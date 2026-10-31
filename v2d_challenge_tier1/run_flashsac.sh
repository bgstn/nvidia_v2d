#!/bin/bash
set -eo pipefail
source /home/jzhang/rz76_scratch2/jzhang/v2d/env.sh

SEQ=${1:-ep013}
STATE=${2:-}            # optional: state_<steps>.safetensors to resume from

cd $REPO/robotic_grounding/flash_chord
source .venv/bin/activate
NV=$(find "$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia" -type d -name lib | paste -sd:)
export LD_LIBRARY_PATH=$NV${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

EP=$REPO/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/whole_body/t2_tier1/sequence_id=${SEQ}/robot_name=g1/data.parquet
RESUME=()
[ -n "$STATE" ] && RESUME=(training.resume=true "training.checkpoint_path=${STATE}")

python scripts/train_flash_sac.py experiment=g1_recon_body_flash_sac \
  "task.parquet='${EP}'" reset.seed=42 logging.mode=disabled \
  output_dir=outputs/g1_${SEQ} "${RESUME[@]}"
