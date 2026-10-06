# Source before running retargeting on a plain GPU server (no SLURM / container):
#   source tools/retarget/local/env.sh
# Layout under B2D_ROOT (default: the parent of this Bench2Dex checkout):
#   Bench2Dex/  IsaacLab/  assets/  dex2bench_dataset -> assets  b2d_origin/dataset/  b2dr_runs/  envs/b2d/
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export B2D_ROOT="${B2D_ROOT:-$(cd "$_here/../../../.." && pwd)}"
export B2DR_RUNS="${B2DR_RUNS:-$B2D_ROOT/b2dr_runs}"
export B2D_ENV="${B2D_ENV:-$B2D_ROOT/envs/b2d}"
_conda="$(conda info --base 2>/dev/null)"
[ -n "$_conda" ] && source "$_conda/etc/profile.d/conda.sh"
conda activate "$B2D_ENV"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export HF_HOME="${HF_HOME:-$B2D_ROOT/.cache/huggingface}" PIP_CACHE_DIR="${PIP_CACHE_DIR:-$B2D_ROOT/.cache/pip}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Isaac Sim's bundled libs need the env's newer libstdc++ (CXXABI_1.3.15)
export LD_PRELOAD="$B2D_ENV/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
[ -s "$HOME/.hf_token" ] && export HF_TOKEN="${HF_TOKEN:-$(tr -d '[:space:]' < "$HOME/.hf_token")}"
unset _here _conda
