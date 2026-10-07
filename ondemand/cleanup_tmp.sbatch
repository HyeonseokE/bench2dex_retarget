#!/bin/bash
# Free the node's /tmp (p1's root disk holds /tmp and is ~99% full): remove THIS user's leftovers that
# no running job needs. Paste in OOD Job Composer and submit; it prints what it found and what it removed.
#
# Removed (node-local copies / caches, recreated on demand by the job that needs them):
#   /tmp/lerobot-ds-$USER     kaia_lerobot training: datasets staged from NFS (re-staged by the next training)
#   /tmp/apptainer-$USER      kaia_lerobot: apptainer layer cache (only used while pulling an image)
#   /tmp/b2d-$USER/apptainer-cache, apptainer-tmp   this pipeline's layer cache (SIF already built)
# Kept: /tmp/b2d-$USER/home (Isaac Sim caches of the running retargeting) and /tmp/b2d-$USER/mps-*.
# The kaia_lerobot items are skipped while any non-b2dr job of this user runs (it may be a training
# that reads its staged dataset). Checkpoints live under $HOME (NAS), not /tmp, so they are untouched.
#
#SBATCH --job-name=b2dr-cleanup
#SBATCH --partition=gpu
#SBATCH --nodelist=p1
#SBATCH --gres=gpu:pro6000:1        # the gpu partition does not schedule 0-GPU jobs; unused
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-b2dr-cleanup-%j.out

set -uo pipefail
export USER="${USER:-$(id -un)}"
echo "=== /tmp on $(hostname -s) before: $(df -h /tmp | tail -1 | awk '{print $4" free, "$5" used"}') ==="
echo "--- this user's entries in /tmp:"
find /tmp -maxdepth 1 -user "$USER" -exec du -sh {} + 2>/dev/null | sort -h

OTHER="$(squeue -h -u "$USER" -o '%i %j' | awk -v me="${SLURM_JOB_ID:-0}" '$1 != me && $2 !~ /^b2dr-/' | wc -l)"
TARGETS=("/tmp/b2d-$USER/apptainer-cache" "/tmp/b2d-$USER/apptainer-tmp")
if [ "$OTHER" -eq 0 ]; then
  TARGETS+=("/tmp/lerobot-ds-$USER" "/tmp/apptainer-$USER")
else
  echo "--- $OTHER other job(s) of $USER running: keeping /tmp/lerobot-ds-$USER and /tmp/apptainer-$USER"
  squeue -u "$USER" -o '%.10i %.20j %.8T'
fi
for d in "${TARGETS[@]}"; do
  [ -e "$d" ] || continue
  echo "removing $d ($(du -sh "$d" 2>/dev/null | cut -f1))"
  rm -rf "$d"
done
echo "=== /tmp after: $(df -h /tmp | tail -1 | awk '{print $4" free, "$5" used"}') ==="
