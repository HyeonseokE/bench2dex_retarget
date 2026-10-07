#!/bin/bash
# Is the retargeting running fine? Run on the login node / OOD shell (not submitted):
#   bash ~/b2d/bench2dex_retarget/ondemand/status.sh
# Shows: the b2dr jobs; progress per (task, target) from the worker's report (refreshed every 2 min);
# the worker log's latest start / finish lines and errors; GPU memory, sessions and MPS on the node;
# free space of the node's /tmp.
REPO="$HOME/b2d/bench2dex_retarget"
REPORT="$REPO/results/report"

echo "=== jobs ==="
squeue -u "$USER" -o '%.10i %.18j %.9T %.12M %.10l %.6D %R' | grep -E 'JOBID|b2dr' || echo "(no b2dr jobs)"

echo; echo "=== progress (results/report/STATUS.md) ==="
if [ -f "$REPORT/STATUS.md" ]; then grep -E '^Updated|^Total|^\|' "$REPORT/STATUS.md"
else echo "(no report yet: the worker writes it once its queue starts)"; fi

W="$(squeue -h -u "$USER" -n b2dr-retarget -t R -o '%i' | head -1)"
if [ -z "$W" ]; then echo; echo "(no running b2dr-retarget worker)"; exit 0; fi
LOG="$(scontrol show job "$W" | grep -o 'StdOut=[^ ]*' | cut -d= -f2)"

echo; echo "=== worker $W log: $LOG ==="
grep -E 'MPS|sessions use|FATAL|WARNING' "$LOG" | head -5
echo "--- latest:"
grep -E '\] (start|stage 1 failed|[0-9]+ ep[0-9]+ -> )' "$LOG" | tail -12
echo "--- finished: $(grep -cE ' -> [a-z0-9]+: ' "$LOG")   recorded: $(grep -cE ' -> [a-z0-9]+: recorded' "$LOG")   failed: $(grep -cE ' -> [a-z0-9]+: (failed_all_attempts|stage2_failed)' "$LOG")"
grep -iE 'Traceback|out of memory|PhysX state' "$LOG" | tail -3

echo; echo "=== node (worker $W) ==="
srun --jobid="$W" --overlap -N1 -n1 bash -c '
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
  echo "sessions on GPUs: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
  pgrep -u "$USER" -f nvidia-cuda-mps-server >/dev/null && echo "MPS server: running" || echo "MPS server: not running"
  df -h /tmp | tail -1 | awk "{print \"node /tmp free: \" \$4 \" (\" \$5 \" used)\"}"
' 2>/dev/null || echo "(srun --overlap not allowed here; check the worker log instead)"
