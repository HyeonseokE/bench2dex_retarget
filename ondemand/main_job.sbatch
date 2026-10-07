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
# Intended use: submit ONCE; the tasks of TASK_QUEUE (edit below) run one after another, one job each,
# all 50 source episodes. Resubmitting is always safe: finished stages and successful episodes are
# skipped (failed ones get new attempts); a job that starts while another b2dr job runs requeues itself
# behind it. To stop the chain, cancel the running job AND the queued one.
#
# Knobs (sbatch --export=ALL,KNOB=value, or edit the defaults below):
#   TASK_QUEUE="12 42 34 60 43 76 08 21 27"  (TASKS="07" runs one task alone)  EPISODES=50  TARGETS="shadow wuji" (default: all but the source)  MAX_ATTEMPTS=5
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
# Tasks run one per job, in this order: this job takes the first, and right at its start queues the
# next job for the rest (--dependency=afterany on this one), so the chain keeps going even if a task fails
# or hits the walltime, and the next job is already waiting in the queue when this one ends.
ALONE=0; [ -n "${TASKS:-}" ] && [ -z "${TASK_QUEUE:-}" ] && ALONE=1      # TASKS=07 alone: no chain
TASK_QUEUE="${TASK_QUEUE:-12 42 34 60 43 76 08 21 27}"
export TASKS="${TASKS:-${TASK_QUEUE%% *}}"
REST="$(echo " $TASK_QUEUE " | sed "s/ $TASKS / /" | xargs)"
[ "$ALONE" = 1 ] && REST=""
export EPISODES="${EPISODES:-50}"
echo "task $TASKS now; queued after it: [${REST:-none}]"

# One b2dr job at a time: if another is running, wait for it (requeue this one behind it) instead of
# racing it for the node and the results tree.
BUSY="$(squeue -h -u "$USER" -t R -o '%i %j' 2>/dev/null | awk -v me="${SLURM_JOB_ID:-0}" '$1 != me && $2 ~ /^b2dr-(main|retarget)/ {print $1}' | head -1)"
if [ -n "$BUSY" ]; then
  SELF="$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | grep -o 'Command=[^ ]*' | cut -d= -f2)"
  [ -r "$SELF" ] || SELF="$0"                      # this very file (the pasted one), not an older checkout
  NEXT="$(sbatch --parsable --chdir="${SLURM_SUBMIT_DIR:-$PWD}" --dependency="afterany:$BUSY" \
          --export=ALL,TASK_QUEUE="$TASK_QUEUE",TASKS="$TASKS" "$SELF")"
  echo "b2dr job $BUSY is running: requeued as job $NEXT after it; exiting"
  exit 0
fi

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
if [ -n "$REST" ]; then      # queue the next task now (from the freshly pulled copy of this file)
  NEXT="$(sbatch --parsable --chdir="${SLURM_SUBMIT_DIR:-$PWD}" --dependency="afterany:${SLURM_JOB_ID}" \
          --export=ALL,TASK_QUEUE="$REST",TASKS="${REST%% *}" "$CODE_DIR/main_job.sbatch")"
  echo "next task ${REST%% *} queued as job $NEXT (runs after this one); then [$(echo " $REST " | sed "s/ ${REST%% *} / /" | xargs)]"
fi

echo "=== [3/6] setup ==="
# shellcheck source=/dev/null
source "$CODE_DIR/setup.sh"

echo "=== [4/6] fetch: tasks [$TASKS] x $EPISODES episodes ==="
export APPTAINERENV_TASKS="$TASKS" APPTAINERENV_EPISODES="$EPISODES"
"${ISAAC[@]}" "$PY" ondemand/fetch_data.py

echo "=== [5/6] retarget (this job, 2 GPUs) ==="
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
