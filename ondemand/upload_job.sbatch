#!/bin/bash
# Upload a finished task's recorded HDF5 to Hugging Face NOW, even if some episodes failed (they are listed
# in the repo's failures.json and can be added later: tools/retarget/upload_task.py counts episodes already on
# the Hub, so a later run that retargeted only the missing ones completes the repo to 50/50).
#
#   OOD Job Composer -> New Job -> paste THIS FILE -> edit TASKS below if needed -> Submit.
#
# Knobs: TASKS="07" (space-separated)  EPISODES=50  TARGETS="" (default: all but the source)
#        ALLOW_PARTIAL=1 (0 = only targets with all EPISODES succeeded)  HF_NAMESPACE (default: token's user)
#
#SBATCH --job-name=b2dr-upload
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1        # unused; the gpu partition does not schedule 0-GPU jobs
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-b2dr-upload-%j.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
TASKS="${TASKS:-07}"
EPISODES="${EPISODES:-50}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-1}"
REPO_DIR="$HOME/b2d/bench2dex_retarget"
# shellcheck source=/dev/null
source "$REPO_DIR/ondemand/env.sh"
sync_repo "" "$REPO_DIR"
export APPTAINERENV_HF_NAMESPACE="${HF_NAMESPACE:-}" APPTAINERENV_HF_STAGES="${HF_STAGES:-origin}"
P=""; [ "$ALLOW_PARTIAL" = "1" ] && P="--allow_partial"
for t in $TASKS; do
  "${ISAAC[@]}" "$PY" tools/retarget/upload_task.py --task "$t" --episodes "$EPISODES" $P \
    ${TARGETS:+--targets "$TARGETS"} || echo "[info] task $t: some targets incomplete (see above)"
done
