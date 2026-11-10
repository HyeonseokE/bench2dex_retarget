#!/bin/bash
# Start the CloudXR 6 runtime (isaacteleop) for Quest 3 / WebXR hand tracking. Keep this terminal open;
# Ctrl+C stops the runtime. Then run tools/teleop_xr/run_quest_teleop.sh in another terminal.
# Ported from DexVerse teleop/quest-cloudxr6 (scripts/teleop_tools/start_cloudxr_runtime.sh).
#
# Pitfalls handled here:
#   * NV_CXR_ENABLE_PUSH_DEVICES=0 selects optical (headset) hand tracking. Without it the runtime picks the
#     "Push Hand Tracker" and every hand joint stays at 0/26.
#   * PYTHONPATH from Isaac Sim / bench2dex_env.sh (websockets 12, pydantic) kills the runtime with
#     "websockets >= 14"; it is cleared before activating the isaacteleop env.
#   * Only one runtime (WSS port 48322) per network namespace.
#   * The media stream goes to the address the runtime advertises. This container has its own network
#     namespace (docker bridge 172.17.x is unreachable from the Quest), so the Tailscale IP is advertised
#     via NV_CXR_ENDPOINT_IP. Override with CXR_ENDPOINT_IP=<ip>.
set -eo pipefail

ENV_PREFIX=${ISAACTELEOP_ENV:-/workspace/envs/isaacteleop}
CXR_HOME=${CXR_HOME:-/workspace/.cloudxr}
ENV_CONFIG=${CXR_ENV_CONFIG:-${CXR_HOME}/cxr_optical.env}
WSS_PORT=48322
TS=$(dirname "${BASH_SOURCE[0]}")/ts

if (exec 3<>"/dev/tcp/127.0.0.1/${WSS_PORT}") 2>/dev/null; then
    echo "[cloudxr] port ${WSS_PORT} is already in use (another CloudXR runtime). Stop it first." >&2
    exit 1
fi
if [ ! -x "${ENV_PREFIX}/bin/python" ]; then
    echo "[cloudxr] ${ENV_PREFIX} missing: run tools/teleop_xr/setup_isaacteleop_env.sh first." >&2
    exit 1
fi

mkdir -p "${CXR_HOME}"
[ -e /root/.cloudxr ] || ln -sfn "${CXR_HOME}" /root/.cloudxr
echo "NV_CXR_ENABLE_PUSH_DEVICES=0" > "${ENV_CONFIG}"

ENDPOINT_IP=${CXR_ENDPOINT_IP:-}
if [ -z "${ENDPOINT_IP}" ]; then
    ENDPOINT_IP=$("${TS}" ip -4 2>/dev/null | head -n 1 || true)
fi
if [ -n "${ENDPOINT_IP}" ]; then
    echo "NV_CXR_ENDPOINT_IP=${ENDPOINT_IP}" >> "${ENV_CONFIG}"
    echo "[cloudxr] media endpoint advertised to the headset: ${ENDPOINT_IP}"
else
    echo "[cloudxr] WARNING: no Tailscale IP (run tools/teleop_xr/setup_tailscale.sh) and CXR_ENDPOINT_IP unset;" >&2
    echo "[cloudxr]          the headset will likely disconnect ~15 s after signing in." >&2
fi

unset PYTHONPATH LD_PRELOAD
source /opt/conda/etc/profile.d/conda.sh
conda activate "${ENV_PREFIX}"

echo "[cloudxr] starting runtime (env config: ${ENV_CONFIG}). The first run asks you to accept the NVIDIA EULA."
exec python -m isaacteleop.cloudxr --cloudxr-env-config="${ENV_CONFIG}" "$@"
