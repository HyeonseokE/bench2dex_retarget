# One-time environment setup, sourced by main_job.sbatch after env.sh. Every step is idempotent and
# stamped, so after the first run (~1 h: a 20 GB SIF build plus pip) this takes seconds.

echo "=== [setup 1/3] Isaac Sim image -> $SIF ==="
if [ -s "$HOME/.ngc_key" ]; then        # nvcr.io/nvidia/isaac-sim is public; a key is only a fallback
  export APPTAINER_DOCKER_USERNAME='$oauthtoken'
  export APPTAINER_DOCKER_PASSWORD="$(tr -d '[:space:]' < "$HOME/.ngc_key")"
fi
(
  flock 9
  if [ -s "$SIF" ]; then
    echo "SIF present ($(du -h "$SIF" | cut -f1))"
  else
    echo "building SIF from $ISAAC_IMAGE (one-off, ~20-40 min) ..."
    apptainer build "$SIF.partial" "$ISAAC_IMAGE" && mv "$SIF.partial" "$SIF"
  fi
) 9>"$SIF.lock"
[ -s "$SIF" ] || { echo "FATAL: SIF build failed"; exit 1; }
# The build leaves ~15 GB of OCI layers in this node's /tmp, never needed again (p1's / was 98% full).
if [ -d "$APPTAINER_CACHEDIR" ] && [ "$(du -s "$APPTAINER_CACHEDIR" | cut -f1)" -gt 2000000 ]; then
  echo "dropping $(du -sh "$APPTAINER_CACHEDIR" | cut -f1) of node-local OCI cache"
  rm -rf "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
  mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
fi

echo "=== [setup 2/3] IsaacLab $ISAACLAB_TAG (Bench2Dex is the synced fork itself) ==="
[ -d "$WS/IsaacLab/.git" ] || GIT clone --branch "$ISAACLAB_TAG" --depth 1 https://github.com/isaac-sim/IsaacLab.git "$WS/IsaacLab"
[ -e "$WS/dex2bench_dataset" ] || ln -s assets "$WS/dex2bench_dataset"
mkdir -p "$WS/assets"

echo "=== [setup 3/3] python env -> $WS/.pyuser ==="
STAMP="$WS/.pyuser/.b2dr_setup_v1.ok"
if [ -f "$STAMP" ]; then
  echo "already set up ($(cat "$STAMP"))"
else
  "${ISAAC[@]}" bash -c '
    set -euo pipefail
    P=/isaac-sim/python.sh
    $P -c "import site; assert site.ENABLE_USER_SITE, \"user site disabled\""
    printf "numpy==1.26.4\ntorch==2.7.0\ntorchvision==0.22.0\npillow==11.3.0\n" > /workspace/.pyuser/constraints.txt
    # Blackwell (sm_120) needs a CUDA 12.8 build; a 2.7.0+cu126 torch imports fine but has no sm_120 kernels.
    $P -c "import torch, sys; sys.exit(0 if torch.__version__.startswith(\"2.7\") and torch.version.cuda == \"12.8\" else 1)" \
      || $P -m pip install --user torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
    # Isaac Lab v2.3.2 pins flatdict==4.0.1, whose setup.py needs pkg_resources: build it against an
    # old setuptools without isolation (IsaacLab issue #4576; failed the first sim check, 2026-10-05).
    $P -m pip install --user "setuptools==65.0.0"
    $P -m pip install --user --no-build-isolation "flatdict==4.0.1"
    $P -m pip install --user -c /workspace/.pyuser/constraints.txt \
      -e /workspace/IsaacLab/source/isaaclab h5py PyYAML scipy trimesh huggingface_hub
  ' || { echo "FATAL: pip setup failed"; exit 1; }
  echo "isaacsim-5.1.0 isaaclab-$ISAACLAB_TAG $(date +%F)" > "$STAMP"
fi
