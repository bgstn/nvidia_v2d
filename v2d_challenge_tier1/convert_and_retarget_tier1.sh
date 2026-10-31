#  unset DISPLAY
#  
#  export REPO=$PWD/video_to_data
#  export SIF=/home/jzhang/rz76_scratch/jzhang/images/isaac-lab_2.3.2.sif   # adjust if your .sif is elsewhere
#  export OVERLAY=/home/jzhang/rz76_scratch/jzhang/images/overlay.img
#  export WORK=/tmp/$USER/v2d
#  export SCRATCH=/home/jzhang/rz76_scratch/jzhang
#  
#  unset DISPLAY
# apptainer exec --nv \
#   --overlay $OVERLAY \
#   --home $WORK/home -B $SCRATCH \
#   -B $REPO:/workspace/video_to_data \
#   $SIF bash $SCRATCH/v2d/install_rg_deps.sh 2>&1 | tee $WORK/install.log
# 
# 
# apptainer exec --nv \
#   --overlay $OVERLAY:ro \
#   --home $WORK/home -B $SCRATCH \
#   -B $REPO:/workspace/video_to_data \
#   $SIF /workspace/isaaclab/isaaclab.sh -p -c \
#   "import numpy, torch, pinocchio, soma, tensordict, robotic_grounding; print('numpy', numpy.__version__, '| all imports OK')"

source env.sh

# convert hugging face video for retargetting
export T1=$SCRATCH/v2d/data/video_to_data_challenge/track_2/tier_1_multiview_caption
rg $SCRATCH/v2d/convert_track2_to_soma.py --tier_dir $T1 --episodes all --subdir t2_tier1 \
  2>&1 | grep -v pthread_setaffinity | tee $OUT/convert_tier1.log

grep -E "ERROR|WARN|treated|Converted" $OUT/convert_tier1.log

# retargetting
cd $REPO/robotic_grounding
BASE=source/robotic_grounding/robotic_grounding/assets/human_motion_data/whole_body/t2_tier1
FPS=30   # use the value the converter printed

for d in $REPO/robotic_grounding/$BASE/sequence_id=ep*; do
  seq=$(basename $d)
  echo "=== $seq ==="
  rg scripts/retarget/soma_to_g1.py $BASE/$seq --soma-subdir t2_tier1 --fps $FPS --save \
    > $OUT/retarget_$seq.log 2>&1 \
    && echo "  ok" || echo "  FAILED (see $OUT/retarget_$seq.log)"
done

ls $REPO/robotic_grounding/$BASE/*/robot_name=g1/*.parquet | wc -l   # should be 30

# replay trajectory on g1
for ep in 000 010 029; do
  rg scripts/replay_motion_video.py \
    --motion_file $BASE/sequence_id=ep$ep/robot_name=g1 \
    --headless --video_out $OUT/t2_ep${ep}_reference.mp4
done












