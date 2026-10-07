#!/bin/bash
# Bench2Dex cross-embodiment SPIDER retargeting -- SINGLE ENTRY POINT.
#
#   OOD Job Composer -> New Job -> paste THIS FILE -> Submit. Nothing else is submitted by hand.
#
# In order:
#   1. preflight  ~/.hf_token must exist (the Hub rate-limits anonymous downloads mid-way)
#   2. git sync   clone or fast-forward $HOME/b2d/bench2dex_retarget, the fork HyeonseokE/bench2dex_retarget (this
#                 pipeline is its ondemand/ folder; Bench2Dex itself comes with it)
#   3. setup      Isaac Sim 5.1 SIF, Isaac Lab v2.3.2 (once; ~1 h the first time)
#   4. fetch      teleop episodes + the assets of the selected tasks (skips what is already there)
#   5. retarget   IN THIS JOB, on the 2 pro6000 GPUs it already holds (no second job, so nobody can take
#                 the GPUs while the assets download): CUDA MPS, PER_GPU (10) sessions per GPU = 20 at once,
#                 through every (episode, target hand) of the task: SPIDER v2 until MetricTracker calls it
#                 a success, recorded as Bench2Dex HDF5 (ondemand/retarget_worker.sbatch -> run_queue.py ->
#                 run_target.py; no rendering on the cluster)
#   6. upload     IN THIS JOB (ondemand/finalize.sbatch): a target's HDF5 goes to
#                 <ns>/b2d-<scene>-<robot>-retargeting ONLY if all EPISODES source episodes succeeded
#
# Live status: ~/b2d/b2dr_status.md (OOD Files app), rewritten every 5 min; summary lines in this log.
#
# Everything after step 2 is read from the pulled fork, so edits to retarget/, tools/retarget/ and
# ondemand/*.sh only need a push. Only an edit to THIS file needs re-pasting.
#
# Intended use: ONE TASK PER SUBMISSION, all 50 source episodes -- TASKS="07" (edit below). Resubmitting
# is always safe: finished stages and successful episodes are skipped (failed ones get new attempts),
# and a second b2dr-main of the same user refuses to start.
#
# Knobs (sbatch --export=ALL,KNOB=value, or edit the defaults below):
#   TASKS="07"  EPISODES=50  TARGETS="shadow wuji" (default: all but the source)  MAX_ATTEMPTS=5
#   PER_GPU=10 (sessions per GPU, 2 GPUs)  MPS_ON=1  SPIDER_ARGS="" (v2)  UPLOAD=1
#   HF_NAMESPACE (default: the token's user)  HF_STAGES="origin"
#
#SBATCH --job-name=b2dr-main
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:2        # held from the start through download, retargeting and upload
#SBATCH --cpus-per-task=32          # 20 sessions x ~1.4 cores (measured)
#SBATCH --mem=192G                  # 20 sessions x ~6.6 GB (measured)
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-b2dr-main-%j.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"
echo "=== b2dr-main start: $(date) on $(hostname) ==="

REPO_URL="${REPO_URL:-https://github.com/HyeonseokE/bench2dex_retarget.git}"   # the fork; this pipeline is its ondemand/
export TASKS="${TASKS:-07}"          # one task per submission (06 12 42 07 34 60 43 76 08 44 21 27)
export EPISODES="${EPISODES:-50}"

echo "=== [1/6] preflight ==="
[ -s "$HOME/.hf_token" ] || { echo "FATAL: ~/.hf_token missing or empty. Create it in the OOD Files app."; exit 1; }
echo "token ok"

echo "=== [2/6] git sync ==="
WS="$HOME/b2d"
B="$WS/bench2dex_retarget"
mkdir -p "$WS"
export APPTAINER_CACHEDIR="/tmp/b2d-$USER/apptainer-cache"
mkdir -p "$APPTAINER_CACHEDIR"
LG() { apptainer exec --bind "$WS" docker://hyeonseoke/lerobot:v1 git "$@"; }
if [ -d "$B/.git" ]; then
  if [ "$(LG -C "$B" remote get-url origin)" != "$REPO_URL" ]; then
    # an earlier setup cloned the authors' repo here, detached at fd90dcc: switch it to the fork
    echo "switching $B to $REPO_URL"
    LG -C "$B" remote set-url origin "$REPO_URL"
    LG -C "$B" fetch -q origin && LG -C "$B" checkout -q -B main origin/main
  else
    LG -C "$B" pull --ff-only || { echo "FATAL: git pull failed (network, or the checkout diverged). Fix it by hand."; exit 1; }
  fi
else
  LG clone "$REPO_URL" "$B"
fi
# shellcheck source=/dev/null
source "$B/ondemand/env.sh"
echo "Bench2Dex fork @ $(GIT -C "$REPO_DIR" rev-parse --short HEAD)"

echo "=== [3/6] setup ==="
# shellcheck source=/dev/null
source "$CODE_DIR/setup.sh"

echo "=== [4/6] fetch: tasks [$TASKS] x $EPISODES episodes ==="
export APPTAINERENV_TASKS="$TASKS" APPTAINERENV_EPISODES="$EPISODES"
"${ISAAC[@]}" "$PY" ondemand/fetch_data.py

echo "=== [5/6] retarget (this job, 2 GPUs) ==="
OTHERS="$(squeue -h -u "$USER" -o '%i %j' 2>/dev/null | awk -v me="${SLURM_JOB_ID:-0}" '$1 != me && $2 ~ /^b2dr-(main|retarget)/' | wc -l)"
if [ "$OTHERS" -gt 0 ]; then
  echo "FATAL: another b2dr-main / b2dr-retarget job is queued or running; stop it first (both would write the same results):"
  squeue -u "$USER" -o '%.12i %.20j %.10T %.12M' | grep -E 'JOBID|b2dr-'
  exit 1
fi
for t in $TASKS; do
  n="$(ls "$WS/b2d_origin/dataset/${t}"*/origin-generalization/episode_*.hdf5 2>/dev/null | wc -l)"
  [ "$n" -ge "$EPISODES" ] || { echo "FATAL: task $t has $n of $EPISODES source episodes locally"; exit 1; }
done
bash "$CODE_DIR/retarget_worker.sbatch" || echo "[warn] retargeting ended with an error (see above); deciding the upload anyway"

echo "=== [6/6] upload (all-or-nothing per target) ==="
if [ "${UPLOAD:-1}" = "1" ]; then
  bash "$CODE_DIR/finalize.sbatch" || echo "[warn] upload step failed"
fi
echo "=== b2dr-main done: $(date) ==="
