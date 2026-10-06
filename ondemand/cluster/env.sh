# Shared environment for every b2dr cluster job. Sourced, never submitted (no #SBATCH).
#
# Layout: $HOME/b2d is bound to /workspace inside the Isaac Sim container, the same tree the dev box
# has, so b2dr's defaults (B2D_ROOT=/workspace) hold on both. The Isaac Sim 5.1 SIF and the
# PYTHONUSERBASE with Isaac Lab are shared with kaia_lerobot/cluster/b2d_sim_check.sbatch.
#
#   $HOME/b2d/
#     Bench2Dex/            the fork HyeonseokE/Bench2Dex (git pull --ff-only at the start of every job);
#                           this pipeline lives in its ondemand/ folder
#     IsaacLab/             v2.3.2                     assets/ (+ dex2bench_dataset -> assets)
#     b2d_origin/dataset/   teleop episodes             b2dr_runs/  outputs
#     .pyuser/              Isaac Lab + deps            sif/isaac-sim_5.1.0.sif

export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"

WS="$HOME/b2d"
REPO_DIR="$WS/Bench2Dex"                         # git root (the fork)
CODE_DIR="$REPO_DIR/ondemand"                    # this pipeline
SIF="${SIF:-$WS/sif/isaac-sim_5.1.0.sif}"
ISAAC_IMAGE="docker://nvcr.io/nvidia/isaac-sim:5.1.0"
GIT_IMAGE="docker://hyeonseoke/lerobot:v1"      # carries git; the bare compute node may not
ISAACLAB_TAG="v2.3.2"
mkdir -p "$WS/sif" "$WS/.pyuser" "$WS/.cache/pip" "$WS/.cache/huggingface" "$WS/b2dr_runs"

# Node-local scratch: apptainer layers and the container $HOME (Kit's shader/texture caches).
NODE="/tmp/b2d-$USER"
NODE_HOME="$NODE/home"
export APPTAINER_CACHEDIR="$NODE/apptainer-cache"
export APPTAINER_TMPDIR="$NODE/apptainer-tmp"
mkdir -p "$NODE_HOME" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# Headless Vulkan through EGL: the GLX ICD fails without a display (dev box: setup_container.sh).
cat > "$NODE_HOME/nvidia_icd.json" <<'EOF'
{ "file_format_version" : "1.0.1", "ICD": { "library_path": "libEGL_nvidia.so.0", "api_version" : "1.4.312" } }
EOF
cat > "$NODE_HOME/10_nvidia.json" <<'EOF'
{ "file_format_version" : "1.0.0", "ICD" : { "library_path" : "libEGL_nvidia.so.0" } }
EOF

export APPTAINERENV_OMNI_KIT_ACCEPT_EULA=YES
export APPTAINERENV_ACCEPT_EULA=Y
export APPTAINERENV_PRIVACY_CONSENT=Y
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_PYTHONUSERBASE=/workspace/.pyuser
export APPTAINERENV_PIP_CACHE_DIR=/workspace/.cache/pip
export APPTAINERENV_HF_HOME=/workspace/.cache/huggingface
export APPTAINERENV_VK_ICD_FILENAMES="$NODE_HOME/nvidia_icd.json"
export APPTAINERENV___EGL_VENDOR_LIBRARY_FILENAMES="$NODE_HOME/10_nvidia.json"
export APPTAINERENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export APPTAINERENV_B2D_ROOT=/workspace
export APPTAINERENV_HF_HUB_DISABLE_PROGRESS_BARS=1         # no tqdm bars in SLURM logs
if [ -s "$HOME/.hf_token" ]; then
  export APPTAINERENV_HF_TOKEN="$(tr -d '[:space:]' < "$HOME/.hf_token")"
fi

GPU_FLAG=(--nv)
if [ "${NVCCLI:-0}" = "1" ]; then
  export NVIDIA_DRIVER_CAPABILITIES=all
  GPU_FLAG=(--nv --nvccli)
fi
# An array, not a function, so `timeout` can run it.
ISAAC=(apptainer exec "${GPU_FLAG[@]}" --writable-tmpfs
       --home "$NODE_HOME" --bind "$WS:/workspace" --pwd /workspace/Bench2Dex/ondemand "$SIF")
PY=/isaac-sim/python.sh
GIT() { apptainer exec --bind "$WS" "$GIT_IMAGE" git "$@"; }

# Fast-forward a checkout; flock so concurrent array tasks do not race on .git.
sync_repo() {   # sync_repo URL DIR
  if [ -d "$2/.git" ]; then
    flock "$2/.git" apptainer exec --bind "$WS" "$GIT_IMAGE" git -C "$2" pull --ff-only \
      || { echo "FATAL: git pull of $2 failed (network, or the checkout diverged). Fix it by hand."; return 1; }
  else
    GIT clone "$1" "$2"
  fi
  echo "$2 @ $(GIT -C "$2" rev-parse --short HEAD)"
}
