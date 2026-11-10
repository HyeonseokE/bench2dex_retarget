#!/bin/bash
# Run Bench2Dex teleop / demo collection with Meta Quest 3 hand tracking against the running CloudXR runtime.
#
#   tools/teleop_xr/run_quest_teleop.sh --task scenes/06_fruit_bowl_loading.yaml            # teleop only
#   tools/teleop_xr/run_quest_teleop.sh --task scenes/06_fruit_bowl_loading.yaml --collect  # record demos
#
# Added unless you pass them yourself: --teleop --teleop-device quest --headless --xr-stream-log 100.
# Use --gui to drop --headless (scene also visible over VNC/X11; the AR session still starts from code).
# Headset buttons with --collect: START = home + record, STOP = home + save, RESET = discard + home.
set -eo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CXR_HOME=${CXR_HOME:-/workspace/.cloudxr}
CXR_ENV=${CXR_ENV:-${CXR_HOME}/run/cloudxr.env}

# cloudxr.env is written before the EULA check, so also require the runtime's WSS port to be listening.
if [ ! -f "${CXR_ENV}" ] || ! (exec 3<>"/dev/tcp/127.0.0.1/${CXR_WSS_PORT:-48322}") 2>/dev/null; then
    echo "[teleop] CloudXR runtime is not running: start tools/teleop_xr/start_cloudxr_runtime.sh first." >&2
    exit 1
fi

ARGS=("$@")
has_arg() {
    local a
    for a in "${ARGS[@]}"; do
        [[ "$a" == "$1" || "$a" == "$1="* ]] && return 0
    done
    return 1
}
headless=1
if has_arg --gui; then
    headless=0
    filtered=()
    for a in "${ARGS[@]}"; do [[ "$a" != "--gui" ]] && filtered+=("$a"); done
    ARGS=("${filtered[@]}")
fi
has_arg --teleop || ARGS+=(--teleop)
has_arg --teleop-device || ARGS+=(--teleop-device quest)
has_arg --xr-stream-log || ARGS+=(--xr-stream-log 100)
if [[ ${headless} == 1 ]] && ! has_arg --headless; then ARGS+=(--headless); fi

source /workspace/bench2dex_env.sh
set -a
source "${CXR_ENV}"
set +a
unset PYTHONPATH

cd "${REPO_ROOT}"
echo "[teleop] python main.py ${ARGS[*]}"
exec python main.py "${ARGS[@]}"
