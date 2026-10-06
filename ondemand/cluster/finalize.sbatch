#!/bin/bash
# After a task's array: write the dataset card (README.md) and failures.json of every target repo.
# Submitted by cluster/main_job.sbatch with --dependency=afterany on the array -- do not paste this one.
# Episodes that never succeeded are listed there as left out, to be handled later.
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
REPO_DIR="$HOME/b2d/Bench2Dex"
# shellcheck source=/dev/null
source "$REPO_DIR/ondemand/cluster/env.sh"
sync_repo "" "$REPO_DIR"
export APPTAINERENV_HF_NAMESPACE="${HF_NAMESPACE:-}"
for t in ${TASKS:?}; do
  "${ISAAC[@]}" "$PY" scripts/hf_card.py --task "$t" ${TARGETS:+--targets "$TARGETS"} || echo "[warn] card for $t failed"
done
