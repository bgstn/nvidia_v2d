source $SCRATCH/v2d/env.sh
export UV_CACHE_DIR=$SCRATCH/v2d/uv_cache
export UV_PYTHON_INSTALL_DIR=$SCRATCH/v2d/uv_python
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH

cd $REPO && git pull                     # FlashCHORD has had updates since your clone
cd $REPO/robotic_grounding/flash_chord
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[train,dev]"
python -c "import jax; print(jax.devices())"    # should list a CUDA device, not CPU
