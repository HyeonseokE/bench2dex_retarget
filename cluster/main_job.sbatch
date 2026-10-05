#!/bin/bash
# Bench2Dex cross-embodiment SPIDER retargeting -- SINGLE ENTRY POINT.
#
#   OOD Job Composer -> New Job -> paste THIS FILE -> Submit. Nothing else is submitted by hand.
#
# In order:
#   1. preflight  ~/.hf_token must exist (the Hub rate-limits anonymous downloads mid-way)
#   2. git sync   clone or fast-forward $HOME/b2d/dex2bench_retarget from GitHub
#   3. setup      Isaac Sim 5.1 SIF, Bench2Dex @ fd90dcc, Isaac Lab v2.3.2 (once; ~1 h the first time)
#   4. fetch      teleop episodes + the assets of the selected tasks (skips what is already there)
#   5. submit     one array task per (task, episode); each retargets that episode onto every
#                 other robot (cluster/retarget_array.sbatch -> cluster/retarget_body.sh)
#
# Everything after step 2 is read from the pulled checkout, so edits to b2dr/, scripts/ and
# cluster/*.sh only need a push. Only an edit to THIS file needs re-pasting.
#
# Default scope: the 12 tasks below x EPISODES=10 episodes x 4 targets = 480 SPIDER runs. One run is
# ~0.5-2 h depending on episode length (400-1200 frames), so this is a multi-day queue on two GPUs;
# raise EPISODES once the first results look right. Resubmitting is always safe: finished stages
# are skipped and a queue already running refuses duplicates.
#
# Knobs (sbatch --export=ALL,KNOB=value, or edit the defaults below):
#   TASKS="06 12 ..."  EPISODES=10  TARGETS="shadow wuji" (default: all but the source)
#   PACK=2 (targets at once per GPU)  MAX_GPUS=2  SPIDER_ARGS="--num_samples 1024"
#
#SBATCH --job-name=b2dr-main
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1        # held only for setup; a 0-GPU job never schedules on this partition
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-b2dr-main-%j.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"
echo "=== b2dr-main start: $(date) on $(hostname) ==="

REPO_URL="${REPO_URL:-https://github.com/HyeonseokE/dex2bench_retarget.git}"
export TASKS="${TASKS:-06 12 42 07 34 60 43 76 08 44 21 27}"
export EPISODES="${EPISODES:-10}"
MAX_GPUS="${MAX_GPUS:-2}"

echo "=== [1/5] preflight ==="
[ -s "$HOME/.hf_token" ] || { echo "FATAL: ~/.hf_token missing or empty. Create it in the OOD Files app."; exit 1; }
echo "token ok"

echo "=== [2/5] git sync ==="
WS="$HOME/b2d"
mkdir -p "$WS"
export APPTAINER_CACHEDIR="/tmp/b2d-$USER/apptainer-cache"
mkdir -p "$APPTAINER_CACHEDIR"
if [ -d "$WS/dex2bench_retarget/.git" ]; then
  apptainer exec --bind "$WS" docker://hyeonseoke/lerobot:v1 git -C "$WS/dex2bench_retarget" pull --ff-only \
    || { echo "FATAL: git pull failed (network, or the checkout diverged). Fix it by hand."; exit 1; }
else
  apptainer exec --bind "$WS" docker://hyeonseoke/lerobot:v1 git clone "$REPO_URL" "$WS/dex2bench_retarget"
fi
# shellcheck source=/dev/null
source "$WS/dex2bench_retarget/cluster/env.sh"
echo "repo @ $(GIT -C "$REPO_DIR" rev-parse --short HEAD)"

echo "=== [3/5] setup ==="
# shellcheck source=/dev/null
source "$REPO_DIR/cluster/setup.sh"

echo "=== [4/5] fetch: tasks [$TASKS] x $EPISODES episodes ==="
export APPTAINERENV_TASKS="$TASKS" APPTAINERENV_EPISODES="$EPISODES"
"${ISAAC[@]}" "$PY" cluster/fetch_data.py

echo "=== [5/5] submit ==="
RUNNING="$(squeue -h -u "$USER" -o '%j' 2>/dev/null | grep -c '^b2dr-retarget' || true)"
if [ "$RUNNING" -gt 0 ]; then
  echo "FATAL: $RUNNING b2dr-retarget job(s) already queued or running. Cancel them first to restart:"
  squeue -u "$USER" -o '%.12i %.20j %.10T %.12M' | grep -E 'JOBID|b2dr-retarget'
  exit 1
fi
WORKLIST="$WS/b2dr_runs/worklist_${SLURM_JOB_ID:-manual}.txt"
: > "$WORKLIST"
for t in $TASKS; do
  d="$(ls -d "$WS/b2d_origin/dataset/${t}"*/origin-generalization 2>/dev/null | head -1)"
  [ -n "$d" ] || { echo "[warn] no episodes for task $t"; continue; }
  for e in $(seq 0 $(( EPISODES - 1 ))); do
    [ -f "$d/episode_$(printf %06d "$e").hdf5" ] && echo "$t $e" >> "$WORKLIST"
  done
done
N="$(wc -l < "$WORKLIST")"
[ "$N" -gt 0 ] || { echo "FATAL: empty worklist"; exit 1; }
echo "$N items -> $WORKLIST"
sbatch --array="0-$(( N - 1 ))%$MAX_GPUS" --export=ALL,WORKLIST="$WORKLIST" "$REPO_DIR/cluster/retarget_array.sbatch"
squeue -u "$USER" -o '%.12i %.20j %.10T %.10M %.20E'
echo "=== b2dr-main done: $(date) ==="
