#!/bin/bash
# Live status of a running retargeting worker -- started BY the worker job in the background (nothing to
# type on the cluster). Every INTERVAL seconds (default 300) it rewrites
#   $HOME/b2d/b2dr_status.md          (open it in the OOD Files app)
# and prints a one-line summary into the worker's SLURM log.
#   monitor.sh WORKER_LOG [INTERVAL]
# Shows: progress per (task, target) (results/report/STATUS.md, written by run_queue.py), the queue's
# latest start / finish lines and errors, GPU memory / utilisation / sessions, MPS server, node /tmp.
LOG="$1"; INTERVAL="${2:-300}"
REPO="$HOME/b2d/bench2dex_retarget"
REPORT="$REPO/results/report/STATUS.md"
OUT="$HOME/b2d/b2dr_status.md"
while true; do
  fin=$(grep -cE '\] [0-9]+ ep[0-9]+ -> [a-z0-9]+: ' "$LOG" 2>/dev/null)
  ok=$(grep -cE '\] [0-9]+ ep[0-9]+ -> [a-z0-9]+: (recorded|rendered|uploaded)' "$LOG" 2>/dev/null)
  bad=$(grep -cE '\] [0-9]+ ep[0-9]+ -> [a-z0-9]+: (failed_all_attempts|stage2_failed)' "$LOG" 2>/dev/null)
  ses=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  mps=$(pgrep -u "$(id -u)" -f nvidia-cuda-mps-server >/dev/null && echo running || echo "not running")
  tmp=$(df -h /tmp | tail -1 | awk '{print $4" free ("$5" used)"}')
  {
    echo "# b2dr retargeting status"
    echo
    echo "updated $(date '+%F %T')  ·  job ${SLURM_JOB_ID:-?} on $(hostname -s)  ·  log \`$LOG\`"
    echo
    echo "HDF5 (successful episodes): \`$REPO/results/dataset/<target>/<scene>/origin-generalization/\`"
    echo
    echo "## items finished: $fin (recorded $ok, failed $bad)"
    echo
    if [ -f "$REPORT" ]; then grep -E '^Total|^\|' "$REPORT"; else echo "(report not written yet)"; fi
    echo
    echo "## node"
    echo '```'
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null
    echo "sessions on GPUs: $ses   MPS server: $mps   /tmp: $tmp"
    echo '```'
    echo
    echo "## latest queue lines"
    echo '```'
    grep -E '\] (start|stage 1 failed|[0-9]+ ep[0-9]+ -> )' "$LOG" 2>/dev/null | tail -15
    echo '```'
    echo
    echo "## errors (last 5)"
    echo '```'
    grep -hiE 'Traceback|out of memory|PhysX state|FATAL' "$LOG" "$REPO"/results/logs/*.log 2>/dev/null | tail -5
    echo '```'
  } > "$OUT.tmp" && mv "$OUT.tmp" "$OUT"
  echo "[monitor $(date +%T)] finished $fin (ok $ok, failed $bad) | sessions $ses | MPS $mps | /tmp $tmp"
  sleep "$INTERVAL"
done
