#!/bin/bash
set -e
PIP="${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir"

${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --upgrade pip
$PIP tensordict==0.8.3 datasets==2.19.0 wandb==0.22.2 importlib_metadata plotly \
     viser==1.0.15 dearpygui scipy scikit-learn tzdata
$PIP py-soma-x==0.2.1
$PIP mujoco "judo-rai>=0.0.5" pin==3.7.0 pin-pink==4.2.0 coal==3.0.1 eigenpy==3.10.3 \
     cmeel==0.57.3 cmeel-boost==1.87.0.1 cmeel-urdfdom==4.0.1 cmeel-console-bridge==1.0.2.3 \
     cmeel-tinyxml2==10.0.0 cmeel-tinyxml==2.6.2.3 cmeel-assimp==5.4.3.1 cmeel-octomap==1.10.0 \
     cmeel-qhull==8.0.2.1 cmeel-zlib==1.3.1 usd-core deprecation
$PIP --upgrade osqp "qpsolvers[clarabel,daqp,proxqp,scs,osqp]==4.9.0"
$PIP onnxruntime-gpu==1.24.4 imageio imageio-ffmpeg pyrender pytorch-ignite pytest \
     boto3 pyzmq msgpack-numpy
$PIP "numpy<2.0.0"
cd /workspace/video_to_data/robotic_grounding
$PIP --no-build-isolation -e ./source/robotic_grounding
echo "=== INSTALL DONE ==="
EOF
