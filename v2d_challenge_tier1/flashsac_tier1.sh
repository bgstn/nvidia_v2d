#!/bin/bash
# Train + evaluate one FlashSAC policy per Track 2 episode, as a Slurm job array.
#
# Submit all 30 Tier 1 episodes, at most 4 running at once:
#   sbatch --array=0-29%4 flashsac_tier1.sh t2_tier1
# Only some episodes:
#   sbatch --array=0,4,7,13 flashsac_tier1.sh t2_tier1
# Tier 2 later:
#   sbatch --array=0-29%4 flashsac_tier1.sh t2_tier2
#
# A policy variant: 2nd argument names it (outputs/<subdir>_<tag>/), the rest are Hydra overrides:
#   sbatch -J sw20 --array=0-29%4 flashsac_tier1.sh t2_tier1 switching20 \
#     training.network.actor_architecture=switching_latent training.network.skill_count=20
#
# Re-evaluate an episode whose eval hit "nefc overflow - please increase njmax":
#   EVAL_NJMAX=2048 sbatch --array=7 flashsac_tier1.sh t2_tier1
#
# Safe to resubmit: finished episodes are skipped, interrupted ones resume from
# their latest state_*.safetensors.
#
# Adjust the #SBATCH lines to your cluster (partition, GPU syntax, account).

#SBATCH --job-name=flashsac
#SBATCH --partition=gpu                # A40 nodes (FlashSAC needs ~46 GB free GPU memory)
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G                     # peak RSS ~15 GB; smaller requests backfill sooner
#SBATCH --time=06:30:00                # longest Tier 1 run took 5 h 18 min; resubmitting resumes
#SBATCH --output=logs_flashsac/%x_%A_%a.out
# #SBATCH --account=rz76

set -uo pipefail

SUBDIR=${1:-t2_tier1}
TAG=${2:-}
EXTRA=("${@:3}")
SEQ=$(printf "ep%03d" "${SLURM_ARRAY_TASK_ID:?run this as an array job}")
FINAL_STEPS=349999104

source /home/jzhang/rz76_scratch2/jzhang/v2d/env.sh     # adjust if your env.sh lives elsewhere
FC=$REPO/robotic_grounding/flash_chord
cd "$FC"
source .venv/bin/activate
NV=$(find "$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia" -type d -name lib | paste -sd:)
export LD_LIBRARY_PATH=$NV${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# Per-job caches so parallel jobs never share compilation caches.
export XDG_CACHE_HOME=/tmp/$USER/flashsac_${SLURM_JOB_ID}/cache
mkdir -p "$XDG_CACHE_HOME"

EP=$REPO/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/whole_body/${SUBDIR}/sequence_id=${SEQ}/robot_name=g1/data.parquet
OUTDIR=outputs/${SUBDIR}${TAG:+_$TAG}/${SEQ}
mkdir -p "$OUTDIR"

echo "Host $(hostname)  Job ${SLURM_JOB_ID}  Task ${SLURM_ARRAY_TASK_ID}  ${OUTDIR}  ${EXTRA[*]}"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

if [[ ! -f "$EP" ]]; then
  echo "SKIP: no retargeted parquet at $EP"
  exit 0
fi

FINAL=$OUTDIR/policy_${FINAL_STEPS}.safetensors

# Ground-only episodes have no support surface (empty stages were moved aside):
# tell FlashCHORD not to look for one.
STAGE=$REPO/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/whole_body/reconstructed_stage
SUPPORT=()
if [[ ! -f "$STAGE/${SEQ}_support.usda" && ! -f "$STAGE/${SEQ}_g1_support.usda" ]]; then
  echo "Ground-only episode: scene.support=false"
  SUPPORT=(scene.support=false)
fi

# ---------------- train (skip if finished, resume if interrupted) ----------------
if [[ -f "$FINAL" ]]; then
  echo "Training already finished: $FINAL"
else
  RESUME=()
  STATE=$(ls -v "$OUTDIR"/state_*.safetensors 2>/dev/null | tail -1 || true)
  if [[ -n "$STATE" ]]; then
    echo "Resuming from $STATE"
    RESUME=(training.resume=true "training.checkpoint_path=${STATE}")
  fi
  python scripts/train_flash_sac.py experiment=g1_recon_body_flash_sac \
    "task.parquet='${EP}'" reset.seed=42 logging.mode=disabled \
    output_dir="$OUTDIR" "${SUPPORT[@]}" "${EXTRA[@]}" "${RESUME[@]}" \
    > "$OUTDIR/train.log" 2>&1
  rc=$?
  echo "Training exit code: $rc"
  if [[ ! -f "$FINAL" ]]; then
    echo "FAILED: no final policy; see $FC/$OUTDIR/train.log"
    tail -20 "$OUTDIR/train.log"
    exit 1
  fi
fi

# ---------------- evaluate ----------------
if [[ -f "$FINAL.metrics.json" ]]; then
  echo "Already evaluated: $FINAL.metrics.json"
else
  EVAL=(scripts/evaluate_policy.py)
  # The checkpoint fixes sim.njmax; the wrapper raises it for contact-heavy episodes.
  [[ -n "${EVAL_NJMAX:-}" ]] && EVAL=(/home/jzhang/rz76_scratch2/jzhang/v2d/evaluate_policy_njmax.py)
  python "${EVAL[@]}" "evaluation.checkpoint=${FINAL}" \
    evaluation.world_count=4096 \
    "evaluation.metrics_output=${FINAL}.metrics.json" \
    "evaluation.object_trajectories_output=${FINAL}.objtraj.npz" \
    > "$OUTDIR/eval.log" 2>&1 || { echo "Evaluation failed; see $FC/$OUTDIR/eval.log"; exit 1; }
fi

python - "$FINAL.metrics.json" << 'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
g = lambda k: m.get("metrics", {}).get(k, float("nan"))
print(f"DONE  AUC={g('add_auc'):.3f}  SP-SR={g('spider_sr_uncentered'):.3f}  "
      f"MP-SR={g('maniptrans_sr'):.3f}  MPPE={g('mppe_cm'):.1f}cm  chord_sr={g('chord_sr'):.3f}")
EOF
