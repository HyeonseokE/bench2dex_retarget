#!/bin/bash
# After a task's array: per target hand, push the recorded HDF5 of all EPISODES source episodes to
# <ns>/b2d-<scene>-<target>-retargeting (+ dataset card) -- ONLY if every one of them succeeded; otherwise
# nothing is pushed and the missing episodes are printed (tools/retarget/upload_task.py). Resubmit
# main_job.sbatch to retry the missing ones; the upload is decided again at its end.
# Run by ondemand/main_job.sbatch at its end (bash, not sbatch); can also be submitted on its own.
#
#SBATCH --job-name=b2dr-finalize
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1        # unused; the gpu partition does not schedule 0-GPU jobs
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-b2dr-finalize-%j.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
REPO_DIR="$HOME/b2d/bench2dex_retarget"
# shellcheck source=/dev/null
source "$REPO_DIR/ondemand/env.sh"
sync_repo "" "$REPO_DIR"
export APPTAINERENV_HF_NAMESPACE="${HF_NAMESPACE:-}" APPTAINERENV_HF_STAGES="${HF_STAGES:-origin}"
for t in ${TASKS:?}; do
  "${ISAAC[@]}" "$PY" tools/retarget/upload_task.py --task "$t" --episodes "${EPISODES:-50}" \
    ${TARGETS:+--targets "$TARGETS"} || echo "[info] task $t: not every target complete, see above"
done
