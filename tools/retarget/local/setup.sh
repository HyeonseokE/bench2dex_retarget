#!/bin/bash
# One-time setup on a plain GPU server (conda, no container), mirroring the dev box:
# Isaac Sim 5.1 (pip) + Isaac Lab v2.3.2 in a Python 3.11 conda env, then the teleop episodes and assets.
#
#   git clone https://github.com/HyeonseokE/Bench2Dex.git $B2D_ROOT/Bench2Dex
#   bash $B2D_ROOT/Bench2Dex/tools/retarget/local/setup.sh                 # env only
#   TASKS="06 12 42" bash .../setup.sh --fetch                             # + data for these tasks
#
# Needs: NVIDIA driver >= 570 (RTX 30xx/40xx/50xx, A/H/L-series), conda, git, ~60 GB disk
# (env ~15 GB, assets ~19 GB, episodes ~4 MB each). Put your Hugging Face token in ~/.hf_token.
# Headless containers also need Vulkan: apt install libvulkan1 and an NVIDIA ICD json pointing to
# libEGL_nvidia.so.0 (see README "로컬 GPU 서버").
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B2D_ROOT="${B2D_ROOT:-$(cd "$HERE/../../../.." && pwd)}"
ENV_DIR="${B2D_ENV:-$B2D_ROOT/envs/b2d}"
ISAACLAB_TAG=v2.3.2
cd "$B2D_ROOT"
mkdir -p assets b2d_origin b2dr_runs .cache
[ -e dex2bench_dataset ] || ln -s assets dex2bench_dataset     # Bench2Dex robot configs load ../dex2bench_dataset/...

source "$(conda info --base)/etc/profile.d/conda.sh"
if [ ! -x "$ENV_DIR/bin/python" ]; then
  echo "=== conda env $ENV_DIR (python 3.11) ==="
  conda create -y -p "$ENV_DIR" python=3.11
fi
conda activate "$ENV_DIR"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PIP_CACHE_DIR="$B2D_ROOT/.cache/pip"
STAMP="$ENV_DIR/.b2dr_local_setup_v1.ok"
if [ ! -f "$STAMP" ]; then
  echo "=== torch 2.7 (cu128: also has Blackwell sm_120 kernels) ==="
  pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
  echo "=== Isaac Sim 5.1 ==="
  pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
  echo "=== Isaac Lab $ISAACLAB_TAG ==="
  [ -d IsaacLab/.git ] || git clone --branch "$ISAACLAB_TAG" --depth 1 https://github.com/isaac-sim/IsaacLab.git IsaacLab
  # flatdict 4.0.1 (pinned by Isaac Lab) needs pkg_resources at build time (IsaacLab issue #4576)
  pip install "setuptools==65.0.0" && pip install --no-build-isolation "flatdict==4.0.1"
  printf "numpy==1.26.4\ntorch==2.7.0\ntorchvision==0.22.0\n" > "$ENV_DIR/constraints.txt"
  pip install -c "$ENV_DIR/constraints.txt" -e IsaacLab/source/isaaclab h5py PyYAML scipy trimesh huggingface_hub
  python -c "import isaacsim, isaaclab, torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  echo "isaacsim-5.1.0 isaaclab-$ISAACLAB_TAG $(date +%F)" > "$STAMP"
fi

if [ "${1:-}" = "--fetch" ]; then
  : "${TASKS:?set TASKS, e.g. TASKS=\"06 12 42 07 34 60 43 76 08 44 21 27\"}"
  [ -s "$HOME/.hf_token" ] && export HF_TOKEN="$(tr -d '[:space:]' < "$HOME/.hf_token")"
  echo "=== data: tasks $TASKS, ${EPISODES:-50} episodes each ==="
  B2D_ROOT="$B2D_ROOT" TASKS="$TASKS" EPISODES="${EPISODES:-50}" python Bench2Dex/ondemand/fetch_data.py
fi
echo "SETUP OK  ->  source $B2D_ROOT/Bench2Dex/tools/retarget/local/env.sh"
