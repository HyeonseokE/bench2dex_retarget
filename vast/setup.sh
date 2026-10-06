#!/bin/bash
# Rebuild the dev-box environment on a vast.ai instance (image nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04).
# Same layout as the dev box: everything under /workspace, conda env /workspace/envs/env_isaaclab
# (Python 3.11, Isaac Sim 5.1.0 pip, torch 2.7.0+cu128, Isaac Lab v2.3.2), activated by
# /workspace/bench2dex_env.sh. Idempotent: finished steps are skipped, so it is safe as an on-start script.
#   bash /workspace/Bench2Dex/vast/setup.sh            # environment only
#   TASKS="06" bash /workspace/Bench2Dex/vast/setup.sh # + teleop episodes and Bench2Dex/Assets (~20 GB)
set -euo pipefail
WS=/workspace
ENV=$WS/envs/env_isaaclab
export PIP_CACHE_DIR=$WS/.cache/pip TMPDIR=$WS/.tmp
mkdir -p $WS/.tmp $WS/.cache/ov $WS/.local_share_ov $WS/envs

echo "=== [1/6] system packages, Vulkan/EGL ICDs (headless rendering) ==="
if ! dpkg -s libvulkan1 >/dev/null 2>&1; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    git git-lfs wget ca-certificates ffmpeg libvulkan1 vulkan-tools libxt6 libglu1-mesa libxrandr2 \
    libxinerama1 libxcursor1 libxi6 libsm6 libice6 libegl1
fi
mkdir -p /usr/share/vulkan/icd.d /usr/share/glvnd/egl_vendor.d
echo '{ "file_format_version" : "1.0.1", "ICD": { "library_path": "libEGL_nvidia.so.0", "api_version" : "1.4.312" } }' \
  > /usr/share/vulkan/icd.d/nvidia_icd.json
echo '{ "file_format_version" : "1.0.0", "ICD" : { "library_path" : "libEGL_nvidia.so.0" } }' \
  > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
mkdir -p /root/.cache /root/.local/share
[ -e /root/.cache/ov ] || ln -s $WS/.cache/ov /root/.cache/ov
[ -e /root/.local/share/ov ] || ln -s $WS/.local_share_ov /root/.local/share/ov

echo "=== [2/6] miniconda ==="
if [ ! -x /opt/conda/bin/conda ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $WS/.tmp/mc.sh
  bash $WS/.tmp/mc.sh -b -p /opt/conda && rm $WS/.tmp/mc.sh
fi
source /opt/conda/etc/profile.d/conda.sh

echo "=== [3/6] conda env $ENV (python 3.11) ==="
if [ ! -x $ENV/bin/python ]; then
  conda create -y -q -p $ENV --override-channels -c conda-forge python=3.11 "libstdcxx-ng>=14"
fi
cat > $WS/bench2dex_env.sh <<'ENVEOF'
# Source this before running anything Bench2Dex-related:  source /workspace/bench2dex_env.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate /workspace/envs/env_isaaclab
export PIP_CACHE_DIR=/workspace/.cache/pip
export TMPDIR=/workspace/.tmp
export HF_HOME=/workspace/.cache/huggingface
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export LD_PRELOAD=/workspace/envs/env_isaaclab/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENVEOF
set +u; source $WS/bench2dex_env.sh; set -u
P=$ENV/bin/python

echo "=== [4/6] Isaac Sim 5.1.0, torch 2.7.0+cu128, Bench2Dex packages ==="
STAMP=$ENV/.b2d_vast_setup_v1.ok
if [ ! -f $STAMP ]; then
  printf "numpy==1.26.4\ntorch==2.7.0\ntorchvision==0.22.0\n" > $WS/.tmp/constraints.txt
  $P -m pip install -q "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
  $P -m pip install -q torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
  # Bench2Dex requirements.txt (pinocchio's PyPI name is "pin") + retargeting/upload extras
  $P -m pip install -q -c $WS/.tmp/constraints.txt numpy==1.26.4 h5py PyYAML usd-core opencv-python Pillow \
    imageio pandas pynput tqdm zstandard Flask matplotlib pin dex-retargeting scipy trimesh huggingface_hub pyarrow
fi

echo "=== [5/6] Isaac Lab v2.3.2 + fork checkout ==="
[ -d $WS/IsaacLab/.git ] || git clone -q --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git $WS/IsaacLab
if [ ! -f $STAMP ]; then
  # Isaac Lab pins flatdict==4.0.1, whose setup.py needs pkg_resources (IsaacLab issue #4576)
  $P -m pip install -q "setuptools<70" && $P -m pip install -q --no-build-isolation "flatdict==4.0.1"
  $P -m pip install -q -c $WS/.tmp/constraints.txt -e $WS/IsaacLab/source/isaaclab
  $P - <<'PYEOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "| arch supported:",
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}" in torch.cuda.get_arch_list())
PYEOF
  echo "isaacsim-5.1.0 torch-2.7.0+cu128 isaaclab-v2.3.2 $(date +%F)" > $STAMP
fi
[ -d $WS/Bench2Dex/.git ] || git clone -q https://github.com/HyeonseokE/Bench2Dex.git $WS/Bench2Dex
(cd $WS/Bench2Dex && git remote get-url upstream >/dev/null 2>&1 || git remote add upstream https://github.com/Bench2Dex/Bench2Dex)

echo "=== [6/6] data (only with TASKS set) ==="
if [ -n "${TASKS:-}" ]; then
  [ -n "${HF_TOKEN:-}" ] || echo "WARNING: HF_TOKEN unset; anonymous Hub downloads get rate-limited"
  B2D_ROOT=$WS $P $WS/Bench2Dex/ondemand/fetch_data.py
fi
echo "=== vast setup done: source /workspace/bench2dex_env.sh ==="
