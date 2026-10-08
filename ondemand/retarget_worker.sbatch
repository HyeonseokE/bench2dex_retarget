#!/bin/bash
# One job, two pro6000 GPUs, CUDA MPS on, PER_GPU (default 10) retargeting sessions per GPU at once.
# Run by ondemand/main_job.sbatch inside its own allocation (bash, not sbatch); can also be submitted on
# its own to resume a task whose data is already downloaded (sbatch --export=ALL,TASKS=07,EPISODES=50 ...).
#
# Progress: $HOME/b2d/b2dr_status.md (OOD Files app), rewritten every 5 min by ondemand/monitor.sh, which
# this job starts; a one-line summary also goes into this job's log.
#
# Runs tools/retarget/run_queue.py inside the Isaac Sim container: a queue of (episode, target hand)
# items of $TASKS, 0..EPISODES-1, onto every hand but the source; each item is tools/retarget/run_target.py
# (stage 2, SPIDER with the v2 defaults and attempt schedule, Bench2Dex HDF5 record; no rendering, no
# upload -- ondemand/finalize.sbatch decides the upload after this job). Stage 1 runs once per episode
# in a free slot. Finished items are skipped, failed ones get new attempts if MAX_ATTEMPTS grew, so
# resubmitting resumes.
#
# MPS: one control daemon for the job's GPUs, pipe/log dirs node-local under /tmp (bound into the
# container). Without nvidia-cuda-mps-control on the node the sessions still run, just without MPS.
#
# Sized from measurements (1024-sample SPIDER: ~1.4 cores, ~6.6 GB RAM, ~5.1 GB VRAM per session):
# 20 sessions ~ 28 cores, ~130 GB RAM, ~51 GB VRAM per GPU. Edit the #SBATCH lines if the node is smaller.
#
#SBATCH --job-name=b2dr-retarget
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-b2dr-retarget-%j.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
REPO_DIR="$HOME/b2d/bench2dex_retarget"
# shellcheck source=/dev/null
source "$REPO_DIR/ondemand/env.sh"
sync_repo "" "$REPO_DIR"
cd "$REPO_DIR"
PER_GPU="${PER_GPU:-10}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"         # cluster: 3 attempts per (episode, target) (user, 2026-10-08)
echo "=== b2dr-retarget: tasks [$TASKS] eps 0..$(( EPISODES - 1 )) on $(hostname), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?} $(date) ==="
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader

# ------------------------------------------------------------------ MPS
MPS_DIR="$NODE/mps-${SLURM_JOB_ID:-$$}"
export CUDA_MPS_PIPE_DIRECTORY="$MPS_DIR/pipe" CUDA_MPS_LOG_DIRECTORY="$MPS_DIR/log"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
MPS=0
if [ "${MPS_ON:-1}" = "1" ] && command -v nvidia-cuda-mps-control >/dev/null; then
  nvidia-cuda-mps-control -d && MPS=1
  trap 'kill ${MON:-} 2>/dev/null; echo quit | nvidia-cuda-mps-control; rm -rf "$MPS_DIR"' EXIT
  export APPTAINERENV_CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" APPTAINERENV_CUDA_MPS_LOG_DIRECTORY="$CUDA_MPS_LOG_DIRECTORY"
  echo "MPS control daemon started ($MPS_DIR)"
else
  echo "WARNING: MPS not started (MPS_ON=${MPS_ON:-1}, nvidia-cuda-mps-control $(command -v nvidia-cuda-mps-control || echo missing)); running without MPS"
fi

# ------------------------------------------------------------------ GPU ids as the sessions see them
# A session pins its GPU with CUDA_VISIBLE_DEVICES=<id>. Depending on cgroups / MPS that id is the
# job-relative ordinal (0 1) or the node id SLURM gave; take the first form that reaches two GPUs.
probe() { "${ISAAC[@]}" "$PY" -c 'import torch; print(torch.cuda.get_device_properties(0).uuid)' 2>/dev/null | tail -1; }
GPUS=""
for cand in "0 1" "$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' ' ')"; do
  u=()
  for g in $cand; do u+=("$(APPTAINERENV_CUDA_VISIBLE_DEVICES=$g probe)"); done
  if [ "${#u[@]}" -eq 2 ] && [ -n "${u[0]}" ] && [ -n "${u[1]}" ] && [ "${u[0]}" != "${u[1]}" ]; then GPUS="$cand"; break; fi
done
[ -n "$GPUS" ] || { echo "FATAL: could not address two distinct GPUs"; exit 1; }
echo "sessions use CUDA_VISIBLE_DEVICES in [$GPUS], $PER_GPU per GPU, MPS=$MPS"
[ "$MPS" = 1 ] && echo "get_server_list" | nvidia-cuda-mps-control || true

# ------------------------------------------------------------------ queue
# progress report (STATUS.md per task x target, refreshed every 2 min) into results/report/ -- not the
# tracked experiments/, so the checkout stays clean for git pull. ondemand/monitor.sh summarises it.
export APPTAINERENV_B2DR_EXPERIMENTS=/workspace/bench2dex_retarget/results/report
bash "$CODE_DIR/outputs_note.sh" || true     # OUTPUTS.md in the job folder: where the HDF5 files go
MYLOG="$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | grep -o 'StdOut=[^ ]*' | cut -d= -f2)"
bash "$CODE_DIR/monitor.sh" "${MYLOG:-/dev/null}" "${MONITOR_S:-300}" &
MON=$!
echo "live status: $HOME/b2d/b2dr_status.md (every ${MONITOR_S:-300} s)"
"${ISAAC[@]}" env PY="$PY" "$PY" tools/retarget/run_queue.py --tasks $TASKS \
  --episodes $(seq 0 $(( EPISODES - 1 ))) ${TARGETS:+--targets $TARGETS} \
  --gpus $GPUS --per_gpu "$PER_GPU" --max_attempts "$MAX_ATTEMPTS" --spider_args "${SPIDER_ARGS:-}" \
  --no_render --no_upload --retry_failed

kill "$MON" 2>/dev/null || true
timeout 20 bash "$CODE_DIR/monitor.sh" "${MYLOG:-/dev/null}" 99999 || true     # final snapshot
echo "=== summary $(date) ==="
for t in $TASKS; do
  "${ISAAC[@]}" "$PY" tools/retarget/upload_task.py --task "$t" --episodes "$EPISODES" ${TARGETS:+--targets "$TARGETS"} --dry_run || true
done
