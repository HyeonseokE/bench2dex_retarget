#!/bin/bash
# Run the authors' policy/ACT/eval_double_env.sh unchanged, with an allocator-only setting
# (no numerical effect) that reduces fragmentation OOMs when two evals share one GPU.
#   bash tools/eval/act_shared_gpu.sh TASK CKPT_DIR [args...]
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec bash "$(cd "$(dirname "$0")/../.." && pwd)/policy/ACT/eval_double_env.sh" "$@"
