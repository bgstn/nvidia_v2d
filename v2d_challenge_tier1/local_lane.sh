#!/bin/bash
# Use this session's GPU as an extra training lane for Slurm job arrays of flashsac_tier1.sh.
#
# Takes episodes from the end of the arrays: an episode is trained here only if its array task
# is still pending, and that task is cancelled first, so the cluster and this lane never train
# the same episode. Episodes the arrays have started, or that already have a final policy, are
# skipped.
#
#   bash local_lane.sh <wait_pid|-> <subdir> <tag> "<job_id>:<first>-<last> ..." [hydra overrides...]
#   bash local_lane.sh - t2_tier1 switching20 "61701910:0-13 61701911:15-29" \
#     training.network.actor_architecture=switching_latent training.network.skill_count=20
#
# If this session ends mid-episode, resubmit that episode: sbatch --array=<N> flashsac_tier1.sh ...

set -uo pipefail
cd "$(dirname "$0")"

WAIT_PID=$1 SUBDIR=$2 TAG=$3 ARRAYS=$4
shift 4
FINAL_STEPS=349999104
FC=video_to_data/robotic_grounding/flash_chord

# Let a lane episode that is already running finish first.
if [[ $WAIT_PID != - ]]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi

# Interleave the arrays from their last episode backwards: 29 13 28 12 ...
read -ra SPECS <<< "$ARRAYS"
declare -a ORDER=()
for ((step = 0; step < 30; step++)); do
  for spec in "${SPECS[@]}"; do
    job=${spec%%:*} range=${spec#*:}
    first=${range%-*} last=${range#*-}
    episode=$((last - step))
    ((episode >= first)) && ORDER+=("$job:$episode")
  done
done

for item in "${ORDER[@]}"; do
  job=${item%%:*} episode=${item#*:}
  seq=$(printf "ep%03d" "$episode")
  [[ -f $FC/outputs/${SUBDIR}_${TAG}/$seq/policy_${FINAL_STEPS}.safetensors.metrics.json ]] && continue
  # Only take episodes whose array task has not started.
  if ! squeue -h -r -j "${job}_${episode}" -t PD -o %i 2>/dev/null | grep -q .; then
    continue
  fi
  scancel "${job}_${episode}" || continue
  echo "$(date '+%F %T') lane: $seq (cancelled ${job}_${episode})"
  SLURM_ARRAY_TASK_ID=$episode SLURM_JOB_ID=lane$episode \
    bash flashsac_tier1.sh "$SUBDIR" "$TAG" "$@" > "logs_flashsac/${TAG}_lane_${episode}.out" 2>&1
  tail -1 "logs_flashsac/${TAG}_lane_${episode}.out"
done
echo "$(date '+%F %T') lane: no pending episodes left"
