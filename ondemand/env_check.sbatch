#!/bin/bash
# Environment check for the whole pipeline on the pro6000 nodes -- SINGLE ENTRY POINT.
#
#   OOD Job Composer -> New Job -> paste THIS FILE -> Submit.
#
# Same bootstrap as main_job.sbatch (git sync, setup, fetch -- task 06, episode 0 only), then every layer
# is exercised briefly, in order, so a failure says which layer:
#
#   L1  cuda        torch in the container sees the GPU, has sm_120 (Blackwell) kernels
#   L2  gfx libs    which NVIDIA graphics libs --nv bound (RTX rendering needs them)       [info]
#   L3  isaacsim    Kit boots headless, a cube falls (CPU physics + render)
#   L4  isaaclab    Isaac Lab + GPU PhysX with 1024 cloned envs (SPIDER's sample count)
#   L4c camera      one RGB frame rendered and saved as PNG
#   L5  bench2dex   Bench2Dex's own main.py, task 06, headless, 300 steps
#   L6  stage1      retarget reference of 06 ep0 (source RH56DFX replayed kinematically)
#   L7  stage2      kinematic retarget 06 ep0 -> Shadow
#   L8  stage3      SPIDER, deliberately tiny (64 samples, 1 iteration, 60 frames) -- plumbing, not success
#   L9  stage4      Bench2Dex record of that run (kept even though it fails)
#   L10 stage5      replay.py --restore-generalization (RGB x6 + TacMap) + GT labels of that episode
#   L11 hf          token identity and repo.write permission (nothing is uploaded)
#
# Outputs go to $HOME/b2d/b2dr_runs_envcheck/ (not the real run tree); per-layer logs in
# $HOME/b2d/b2dr_runs_envcheck/checks/<jobid>/, plus the camera PNG. The first run also builds the SIF
# and the Python env and downloads the assets (~1-2 h); later runs only check (~30-60 min).
#
# Knobs: NVCCLI=1 (bind GPU libs by capability if L2 shows them missing), ONLY="L1 L3 ..." (subset).
#
#SBATCH --job-name=b2dr-envcheck
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-b2dr-envcheck-%j.out

set -euo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"
echo "=== b2dr-envcheck start: $(date) on $(hostname) ==="

REPO_URL="${REPO_URL:-https://github.com/HyeonseokE/bench2dex_retarget.git}"   # the fork; this pipeline is its ondemand/
ONLY="${ONLY:-L1 L2 L3 L4 L4c L5 L6 L7 L8 L9 L10 L11}"

echo "=== [L0] host ==="
echo "SLURM_JOB_GRES=${SLURM_JOB_GRES:-<unset>}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null || echo "(no nvidia-smi on the node)"
apptainer --version
df -h /tmp "$HOME" | sed 's/^/  /'
L0_INFO="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 || echo '?')"
[ -s "$HOME/.hf_token" ] || { echo "FATAL: ~/.hf_token missing or empty (needed for the downloads). Create it in the OOD Files app."; exit 1; }

echo "=== git sync ==="
WS="$HOME/b2d"
B="$WS/bench2dex_retarget"
mkdir -p "$WS"
export APPTAINER_CACHEDIR="/tmp/b2d-$USER/apptainer-cache"
mkdir -p "$APPTAINER_CACHEDIR"
LG() { apptainer exec --bind "$WS" docker://hyeonseoke/lerobot:v1 git "$@"; }
if [ -d "$B/.git" ]; then
  if [ "$(LG -C "$B" remote get-url origin)" != "$REPO_URL" ]; then
    # an earlier setup cloned the authors' repo here, detached at fd90dcc: switch it to the fork
    echo "switching $B to $REPO_URL"
    LG -C "$B" remote set-url origin "$REPO_URL"
    LG -C "$B" fetch -q origin && LG -C "$B" checkout -q -B main origin/main
  else
    LG -C "$B" pull --ff-only || { echo "FATAL: git pull failed (network, or the checkout diverged). Fix it by hand."; exit 1; }
  fi
else
  LG clone "$REPO_URL" "$B"
fi
# shellcheck source=/dev/null
source "$B/ondemand/env.sh"
echo "Bench2Dex fork @ $(GIT -C "$REPO_DIR" rev-parse --short HEAD)"

echo "=== setup ==="
# shellcheck source=/dev/null
source "$CODE_DIR/setup.sh"

echo "=== fetch (task 06, episode 0, all assets) ==="
export APPTAINERENV_TASKS="06" APPTAINERENV_EPISODES="1"
"${ISAAC[@]}" "$PY" ondemand/fetch_data.py

# ------------------------------------------------------------------ checks
RUNS="$WS/b2dr_runs_envcheck"
OUT="$RUNS/checks/${SLURM_JOB_ID:-manual-$$}"
mkdir -p "$OUT"
export APPTAINERENV_B2DR_RUNS=/workspace/b2dr_runs_envcheck
COUT="/workspace/b2dr_runs_envcheck/checks/$(basename "$OUT")"
declare -A RES INFO
run_check() {   # run_check NAME MARKER TIMEOUT_S cmd...
  local name="$1" marker="$2" tmo="$3"; shift 3
  local log="$OUT/$name.log"
  if [[ " $ONLY " != *" $name "* ]]; then RES[$name]="skip"; return; fi
  echo "=== [$name] $(date +%T) -> $log ==="
  local t0=$SECONDS rc=0
  timeout --kill-after=30 "$tmo" "$@" > "$log" 2>&1 || rc=$?
  INFO[$name]="$(( SECONDS - t0 ))s"
  if grep -q "$marker" "$log"; then RES[$name]="PASS"
  elif [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then RES[$name]="FAIL (timeout ${tmo}s)"
  else RES[$name]="FAIL (rc=$rc)"; fi
  echo "--- $name: ${RES[$name]} in ${INFO[$name]}"
  grep -E "^RESULT|^\[L[0-9]|^L[0-9]|Traceback|Error:|FATAL|Episode step limit|rt_ratio" "$log" | tail -8 | sed 's/^/  /' || true
}
CK=/workspace/bench2dex_retarget/ondemand/checks
EP0="$RUNS/06_fruit_bowl_loading/ep000"

run_check L1 L1_OK 600 "${ISAAC[@]}" "$PY" $CK/l1_cuda.py
run_check L2 L2_DONE 300 "${ISAAC[@]}" bash -c '
  for l in libEGL_nvidia.so.0 libGLX_nvidia.so.0 libnvidia-gpucomp.so libnvidia-rtcore.so libnvoptix.so.1; do
    ls /.singularity.d/libs/$l* >/dev/null 2>&1 && echo "  [have] $l" || echo "  [MISSING] $l"
  done; echo L2_DONE'
run_check L3 L3_OK 1500 "${ISAAC[@]}" "$PY" $CK/l3_isaacsim.py
run_check L4 L4_OK 1500 "${ISAAC[@]}" "$PY" $CK/l4_camera.py --headless --num_envs 1024
run_check L4c L4C_OK 1200 "${ISAAC[@]}" "$PY" $CK/l4_camera.py --headless --enable_cameras --num_envs 1 --png "$COUT/l4c_camera.png"
run_check L5 "Episode step limit reached" 2400 "${ISAAC[@]}" bash -c \
  "cd /workspace/bench2dex_retarget && $PY main.py --task scenes/06_fruit_bowl_loading.yaml --episode-steps 300 --headless --profile"
run_check L6 "RESULT stage1" 3600 "${ISAAC[@]}" "$PY" tools/retarget/stage1_reference.py --task 06 --episode 0 --headless
run_check L7 "RESULT stage2" 3600 "${ISAAC[@]}" "$PY" tools/retarget/stage2_kinematic.py --task 06 --episode 0 --target shadow --headless
run_check L8 "RESULT stage3" 3600 "${ISAAC[@]}" "$PY" tools/retarget/stage3_spider.py --task 06 --episode 0 --target shadow \
  --tag envcheck --num_samples 64 --iters 1 --max_frames 60 --headless
run_check L9 "RESULT stage4" 1800 "${ISAAC[@]}" "$PY" tools/retarget/stage4_record.py --task 06 --episode 0 --target shadow \
  --tag envcheck --keep_failed
H5="$(grep -o '"episode_hdf5": "[^"]*"' "$EP0/shadow/envcheck_record.json" 2>/dev/null | cut -d'"' -f4 || true)"
run_check L10 "RESULT stage5" 5400 "${ISAAC[@]}" bash -c \
  "PY=$PY bash tools/retarget/stage5_render.sh \$(dirname '$H5') \$(basename '$H5')"
run_check L11 HF_OK 300 "${ISAAC[@]}" "$PY" -c '
from huggingface_hub import HfApi
w = HfApi().whoami(); fg = w.get("auth", {}).get("accessToken", {}).get("fineGrained") or {}
user = [s for s in fg.get("scoped", []) if s.get("entity", {}).get("name") == w["name"]]
can = (w.get("auth", {}).get("accessToken", {}).get("role") == "write") or any("repo.write" in s.get("permissions", []) for s in user)
print("hf user", w["name"], "| repo.write on own namespace:", can); print("HF_OK" if can else "HF_NO_WRITE")'

# ------------------------------------------------------------------ summary
note() { grep -hE "$2" "$OUT/$1.log" 2>/dev/null | tail -1 | sed 's/^ *//' | cut -c1-100; }
echo
echo "=============================== b2dr-envcheck summary ==============================="
printf "%-4s %-22s %-7s %s\n" L0 host info "$L0_INFO"
for k in L1 L2 L3 L4 L4c L5 L6 L7 L8 L9 L10 L11; do
  case $k in
    L1) n="$(note L1 '^device')";; L2) n="$(grep -c MISSING "$OUT/L2.log" 2>/dev/null || true) graphics libs missing";;
    L3) n="$(note L3 'cube z')";; L4) n="$(note L4 'env-steps/s')";; L4c) n="$(note L4c 'rgb ')";;
    L5) n="$(note L5 'rt_ratio|Episode step limit')";; L11) n="$(note L11 'hf user')";;
    *) n="$(note $k '^RESULT')";;
  esac
  printf "%-4s %-22s %-7s %s\n" "$k" "${RES[$k]}" "${INFO[$k]:-}" "$n"
done
echo "logs: $OUT   camera frame: $OUT/l4c_camera.png   rendered episode: ${H5:-<none>}"
echo "======================================================================================"
for k in L1 L3 L4 L5 L6 L7 L8 L9 L10 L11; do [[ "${RES[$k]}" == FAIL* ]] && exit 1; done
exit 0
