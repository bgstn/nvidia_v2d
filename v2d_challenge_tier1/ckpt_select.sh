#!/bin/bash
# Score earlier checkpoints of finished Tier 1 runs, to choose each episode's policy by local metrics.
#
# One array task per episode. For every run directory given (default: the baseline and switching20),
# evaluates policy_<step>.safetensors for each step in STEPS with the unmodified evaluate_policy.py
# and writes outputs/ckpt_select/<run>/epNNN/policy_<step>.json. Finished files are skipped, so a
# preempted task resumes where it stopped. Runs whose final policy does not exist yet are skipped.
#
#   sbatch -J cksel --array=0-29 ckpt_select.sh
#   sbatch -J cksel --array=0-29 ckpt_select.sh t2_tier1 t2_tier1_switching20

#SBATCH --gres=gpu:1
#SBATCH --constraint=A100-80G|L40S|A40
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs_flashsac/%x_%A_%a.out

set -uo pipefail

RUNS=("$@")
[[ $# -eq 0 ]] && RUNS=(t2_tier1 t2_tier1_switching20)
STEPS=${STEPS:-"300011520 310011904 320012288 330012672 340013056"}
WORLDS=${WORLDS:-1024}
FINAL_STEPS=349999104
SEQ=$(printf "ep%03d" "${SLURM_ARRAY_TASK_ID:?run this as an array job}")

source /home/jzhang/rz76_scratch2/jzhang/v2d/env.sh
cd "$REPO/robotic_grounding/flash_chord"
source .venv/bin/activate
NV=$(find "$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia" -type d -name lib | paste -sd:)
export LD_LIBRARY_PATH=$NV${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export XDG_CACHE_HOME=/tmp/$USER/cksel_${SLURM_JOB_ID}/cache
mkdir -p "$XDG_CACHE_HOME"
echo "Host $(hostname)  Job ${SLURM_JOB_ID}  Task ${SLURM_ARRAY_TASK_ID}  ${RUNS[*]}"

for run in "${RUNS[@]}"; do
  dir=outputs/$run/$SEQ
  [[ -f $dir/policy_${FINAL_STEPS}.safetensors ]] || { echo "$run $SEQ: training not finished"; continue; }
  out=outputs/ckpt_select/$run/$SEQ
  mkdir -p "$out"
  for step in $STEPS; do
    [[ -f $out/policy_${step}.json ]] && continue
    [[ -f $dir/policy_${step}.safetensors ]] || { echo "$run $SEQ: no policy_${step}"; continue; }
    python scripts/evaluate_policy.py "evaluation.checkpoint=$dir/policy_${step}.safetensors" \
      "evaluation.world_count=$WORLDS" "evaluation.metrics_output=$out/policy_${step}.json" \
      > "$out/policy_${step}.log" 2>&1 || echo "$run $SEQ $step: evaluation failed"
    [[ -f $out/policy_${step}.json ]] && python -c "
import json, sys
m = json.load(open(sys.argv[1]))['metrics']
print(f\"$run $SEQ $step  AUC={m['add_auc']:.3f}  SP-SR={m['spider_sr_uncentered']:.3f}  \"
      f\"MP-SR={m['maniptrans_sr']:.3f}  MPPE={m['mppe_cm']:.1f}cm\")" "$out/policy_${step}.json"
  done
done
