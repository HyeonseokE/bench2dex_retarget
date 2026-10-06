#!/bin/bash
# One array task = one (task, episode) line of $WORKLIST, retargeted onto every other robot.
# Submitted by cluster/main_job.sbatch from the pulled checkout -- do not paste this one.
#
# PACK targets share this task's GPU (default 2 of the 4): a 1024-sample SPIDER run keeps the
# 96 GB pro6000 well below full, and the per-step Python loop leaves compute idle between launches.
# Whether 2 beats 1 is not measured yet; compare the per-target "minutes" in spider.json.
#
#SBATCH --job-name=b2dr-retarget
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00            # 4 targets / PACK 2 = 2 rounds of <= ~18 h (1200-frame episodes)
#SBATCH --output=slurm-b2dr-retarget-%A_%a.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
REPO_DIR="$HOME/b2d/Bench2Dex"
# shellcheck source=/dev/null
source "$REPO_DIR/ondemand/cluster/env.sh"
sync_repo "" "$REPO_DIR"                     # pick up pushes made since the entry point ran

[ -n "${SLURM_ARRAY_TASK_ID:-}" ] || { echo "FATAL: not an array task"; exit 1; }
[ -f "${WORKLIST:-}" ] || { echo "FATAL: WORKLIST '$WORKLIST' not found"; exit 1; }
read -r TASK EPISODE < <(sed -n "$(( SLURM_ARRAY_TASK_ID + 1 ))p" "$WORKLIST")
[ -n "${TASK:-}" ] || { echo "FATAL: index $SLURM_ARRAY_TASK_ID past the end of $WORKLIST"; exit 1; }
export TASK EPISODE
echo "=== array $SLURM_ARRAY_TASK_ID: task $TASK episode $EPISODE on $(hostname) $(date) ==="
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true

# shellcheck source=/dev/null
source "$CODE_DIR/cluster/retarget_body.sh"
