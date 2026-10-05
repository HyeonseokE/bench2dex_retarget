#!/bin/bash
# Dev-box equivalent of one cluster array task: every stage of one episode, sequentially.
#
#   bash scripts/run_episode.sh 06 0 shadow [extra stage-3 args...]
#
# Uses the conda env from /workspace/bench2dex_env.sh. Outputs go to $B2DR_RUNS (default
# /workspace/b2dr_runs); every stage skips when its output exists.
set -e
TASK=$1 EP=$2 TGT=$3
shift 3
source /workspace/bench2dex_env.sh >/dev/null 2>&1
cd "$(dirname "$0")/.."
python scripts/stage1_reference.py --task "$TASK" --episode "$EP" --headless
python scripts/stage2_kinematic.py --task "$TASK" --episode "$EP" --target "$TGT" --headless
python scripts/stage3_spider.py --task "$TASK" --episode "$EP" --target "$TGT" --headless "$@"
python scripts/stage4_export.py --task "$TASK" --episode "$EP" --target "$TGT"
