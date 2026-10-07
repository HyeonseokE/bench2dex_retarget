#!/bin/bash
# Write OUTPUTS.md (where this job's results are on the cluster, with counts) into the job's folder.
# Called by retarget_worker.sbatch at its start and by finalize.sbatch at the end; TASKS from the job.
#   outputs_note.sh [DIR]   (default: $SLURM_SUBMIT_DIR, the OOD Job Composer folder of the job)
DIR="${1:-${SLURM_SUBMIT_DIR:-.}}"
R="$HOME/b2d/bench2dex_retarget/results"
{
  echo "# b2dr outputs on the cluster"
  echo
  echo "job ${SLURM_JOB_ID:-?} · written $(date '+%F %T') · tasks [${TASKS:-?}]"
  echo
  echo "| what | path |"
  echo "|---|---|"
  echo "| **retargeted HDF5 (successful episodes only; what is uploaded)** | \`$R/dataset/<target>/<scene>/origin-generalization/episode_NNNNNN.hdf5\` |"
  echo "| per (episode, target) work: status.json, run.log, spider_aK.json / _trace.pkl.gz | \`$R/<scene>/epNNN/<target>/\` |"
  echo "| stage-1 reference per episode | \`$R/<scene>/epNNN/reference.npz\` |"
  echo "| session logs | \`$R/logs/\` |"
  echo "| progress report | \`$R/report/STATUS.md\` · live: \`$HOME/b2d/b2dr_status.md\` |"
  echo "| source episodes | \`$HOME/b2d/b2d_origin/dataset/<scene>/origin-generalization/\` |"
  echo
  echo "## HDF5 so far"
  echo
  found=0
  for d in "$R"/dataset/*/*/origin-generalization; do
    [ -d "$d" ] || continue
    found=1
    echo "- \`$d\`: $(ls "$d" | grep -c '^episode_.*\.hdf5$') episodes"
  done
  [ "$found" = 1 ] || echo "(none yet)"
} > "$DIR/OUTPUTS.md.tmp" && mv "$DIR/OUTPUTS.md.tmp" "$DIR/OUTPUTS.md"
echo "outputs note: $DIR/OUTPUTS.md"
