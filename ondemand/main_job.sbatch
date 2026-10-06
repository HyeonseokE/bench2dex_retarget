#!/bin/bash
# Bench2Dex cross-embodiment SPIDER retargeting -- SINGLE ENTRY POINT.
#
#   OOD Job Composer -> New Job -> paste THIS FILE -> Submit. Nothing else is submitted by hand.
#
# In order:
#   1. preflight  ~/.hf_token must exist (the Hub rate-limits anonymous downloads mid-way)
#   2. git sync   clone or fast-forward $HOME/b2d/Bench2Dex, the fork HyeonseokE/Bench2Dex (this
#                 pipeline is its ondemand/ folder; Bench2Dex itself comes with it)
#   3. setup      Isaac Sim 5.1 SIF, Isaac Lab v2.3.2 (once; ~1 h the first time)
#   4. fetch      teleop episodes + the assets of the selected tasks (skips what is already there)
#   5. submit     one array task per (task, episode); each retargets that episode onto every
#                 other robot until it succeeds, records + renders it like the released data and
#                 pushes it to <ns>/b2d-<scene>-<robot>-retargeting (ondemand/retarget_array.sbatch ->
#                 ondemand/retarget_body.sh -> tools/retarget/run_target.py); then, after the array,
#                 ondemand/finalize.sbatch writes each repo's dataset card and failures.json
#
# Everything after step 2 is read from the pulled fork, so edits to retarget/, tools/retarget/ and
# ondemand/*.sh only need a push. Only an edit to THIS file needs re-pasting.
#
# Intended use: ONE TASK PER SUBMISSION, all 50 source episodes -- TASKS="06" (edit below). The four
# target repos of that task then fill episode by episode (one Hub commit per episode). Resubmitting is
# always safe: finished stages and pushed episodes are skipped, and a queue already running refuses
# duplicates.
#
# Knobs (sbatch --export=ALL,KNOB=value, or edit the defaults below):
#   TASKS="06"  EPISODES=50  TARGETS="shadow wuji" (default: all but the source)  MAX_ATTEMPTS=5
#   PACK=2 (targets at once per GPU)  MAX_GPUS=2  SPIDER_ARGS=""  UPLOAD=1
#   HF_NAMESPACE (default: the token's user)  HF_STAGES="origin replay"
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

REPO_URL="${REPO_URL:-https://github.com/HyeonseokE/Bench2Dex.git}"   # the fork; this pipeline is its ondemand/
export TASKS="${TASKS:-06}"          # one task per submission (06 12 42 07 34 60 43 76 08 44 21 27)
export EPISODES="${EPISODES:-50}"
MAX_GPUS="${MAX_GPUS:-2}"

echo "=== [1/5] preflight ==="
[ -s "$HOME/.hf_token" ] || { echo "FATAL: ~/.hf_token missing or empty. Create it in the OOD Files app."; exit 1; }
echo "token ok"

echo "=== [2/5] git sync ==="
WS="$HOME/b2d"
B="$WS/Bench2Dex"
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

echo "=== [3/5] setup ==="
# shellcheck source=/dev/null
source "$CODE_DIR/setup.sh"

echo "=== [4/5] fetch: tasks [$TASKS] x $EPISODES episodes ==="
export APPTAINERENV_TASKS="$TASKS" APPTAINERENV_EPISODES="$EPISODES"
"${ISAAC[@]}" "$PY" ondemand/fetch_data.py

echo "=== [5/5] submit ==="
RUNNING="$(squeue -h -u "$USER" -o '%j' 2>/dev/null | grep -c '^b2dr-retarget' || true)"
# (one task at a time: finish or cancel the previous task's queue before submitting the next)
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
ARRAY="$(sbatch --parsable --array="0-$(( N - 1 ))%$MAX_GPUS" --export=ALL,WORKLIST="$WORKLIST" "$CODE_DIR/retarget_array.sbatch")"
echo "array job $ARRAY"
if [ "${UPLOAD:-1}" = "1" ]; then
  sbatch --dependency="afterany:${ARRAY%%;*}" --export=ALL "$CODE_DIR/finalize.sbatch"
fi
squeue -u "$USER" -o '%.12i %.20j %.10T %.10M %.20E'
echo "=== b2dr-main done: $(date) ==="
