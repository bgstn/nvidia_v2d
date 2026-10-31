export REPO=$PWD/video_to_data
export SIF=/home/jzhang/rz76_scratch/jzhang/images/isaac-lab_2.3.2.sif   # adjust if your .sif is elsewhere
export OVERLAY=/home/jzhang/rz76_scratch/jzhang/images/overlay.img
export WORK=/tmp/$USER/v2d
export SCRATCH=/home/jzhang/rz76_scratch/jzhang
export HF_HOME=$SCRATCH/v2d/hf_cache     # cache on scratch, not your 16 GB home
export PATH=$HOME/.local/bin:$PATH       # so the `hf` command is found


NV=/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/nvidia
export ORT_LIBPATH=$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/curand/lib:$NV/cuda_runtime/lib:$NV/cuda_nvrtc/lib:$NV/nvjitlink/lib


# Disposable, fast, per-job: container home + caches
export WORK=/tmp/$USER/v2d
# Persistent: logs, videos, test outputs you want to keep
export OUT=$SCRATCH/v2d/runs

rg() {
  unset DISPLAY
  local EXT_FIXED=/tmp/$USER/v2d/home/.local/share/ov/data   # path the overlay's links point to
  mkdir -p $WORK/home/.local/share/ov/data $EXT_FIXED
  local ovl="$OVERLAY:ro"
  [ -n "$RG_RW" ] && ovl="$OVERLAY"
  local binds=(-B $SCRATCH -B $REPO:/workspace/video_to_data
               -B $SCRATCH/v2d/ov_data:$EXT_FIXED)
  [ "$WORK/home/.local/share/ov/data" != "$EXT_FIXED" ] && \
    binds+=(-B $SCRATCH/v2d/ov_data:$WORK/home/.local/share/ov/data)
  local gpucomp=$(ls /usr/lib64/libnvidia-gpucomp.so.* 2>/dev/null | head -1)
  [ -n "$gpucomp" ] && binds+=(-B $gpucomp:/.singularity.d/libs/$(basename $gpucomp))
    env -u VIRTUAL_ENV -u CONDA_PREFIX -u PYTHONHOME -u PYTHONPATH apptainer exec --nv \
    --overlay $ovl \
    --home $WORK/home "${binds[@]}" \
    --pwd /workspace/video_to_data/robotic_grounding \
    --env ACCEPT_EULA=Y,PRIVACY_CONSENT=Y,OMNI_KIT_ACCEPT_EULA=YES,PYTHONUNBUFFERED=1,WANDB_MODE=disabled \
    --env LD_LIBRARY_PATH=$ORT_LIBPATH \
    $SIF /workspace/isaaclab/isaaclab.sh -p "$@"
}
