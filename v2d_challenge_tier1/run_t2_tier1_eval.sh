#!/bin/bash
# Official Track 2 Tier 1 trajectories: kit's unmodified evaluator, 1 world per episode.
set -euo pipefail
V2D=/home/jzhang/rz76_scratch2/jzhang/v2d
FC=$V2D/video_to_data/robotic_grounding/flash_chord
source "$FC/.venv/bin/activate"
NV=$(find "$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia" -type d -name lib | paste -sd:)
export LD_LIBRARY_PATH=$NV${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cd $V2D/submission/v2d_submission_kit
python examples/run_policy_evaluation.py \
  --checkpoints $V2D/submission/t2_tier1_checkpoints.json \
  --results $V2D/submission/t2_tier1_results \
  --runtime-root "$FC" --world-count 1
