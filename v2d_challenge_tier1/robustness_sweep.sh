#!/bin/bash
# Score checkpoints under a grid of perturbed physics (friction, object mass, object size) with
# scripts/evaluate_robustness.py. Results: outputs/robustness/<label>/<setting>.json; finished
# settings are skipped, so a rerun resumes.
#
#   sbatch -J robust robustness_sweep.sh base_anneal=outputs/t2_tier1_objanneal/ep023/policy_449998848.safetensors ...
#   bash robustness_sweep.sh label=checkpoint ...          # on this machine's GPU

#SBATCH --gres=gpu:1
#SBATCH --constraint=A100-80G|L40S|A40
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs_flashsac/%x_%j.out

set -uo pipefail
WORLDS=${WORLDS:-1024}
# friction object_mass object_scale
SETTINGS=(
  "1.0 1.0 1.0"
  "0.5 1.0 1.0" "0.75 1.0 1.0" "1.25 1.0 1.0" "1.5 1.0 1.0"
  "1.0 0.5 1.0" "1.0 1.5 1.0" "1.0 2.0 1.0"
  "1.0 1.0 0.9" "1.0 1.0 1.1"
  "0.5 2.0 1.0" "0.75 1.5 1.0"
)

source /home/jzhang/rz76_scratch2/jzhang/v2d/env.sh
cd "$REPO/robotic_grounding/flash_chord"
source .venv/bin/activate
NV=$(find "$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia" -type d -name lib | paste -sd:)
export LD_LIBRARY_PATH=$NV${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export XDG_CACHE_HOME=/tmp/$USER/robust_${SLURM_JOB_ID:-local}/cache
mkdir -p "$XDG_CACHE_HOME"

for spec in "$@"; do
  label=${spec%%=*} checkpoint=${spec#*=}
  for setting in "${SETTINGS[@]}"; do
    read -r friction mass scale <<< "$setting"
    out=outputs/robustness/$label/f${friction}_m${mass}_s${scale}.json
    [[ -f $out ]] && continue
    mkdir -p "$(dirname "$out")"
    python scripts/evaluate_robustness.py "$checkpoint" --friction "$friction" --object-mass "$mass" \
      --object-scale "$scale" --world-count "$WORLDS" -o "$out" > "${out%.json}.log" 2>&1 \
      || { echo "$label $setting: failed, see ${out%.json}.log"; continue; }
    python -c "
import json, sys
m = json.load(open(sys.argv[1]))['metrics']
print(f\"$label f=$friction m=$mass s=$scale  AUC={m['add_auc']:.3f}  SP-SR={m['spider_sr_uncentered']:.3f}  \"
      f\"MP-SR={m['maniptrans_sr']:.3f}  MPPE={m['mppe_cm']:.1f}cm\")" "$out"
  done
done
