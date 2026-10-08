#!/bin/bash
# One-time setup of the CloudXR 6 runtime (bundled with the isaacteleop pip package) for Quest 3 teleop.
#
#   bash tools/teleop_xr/setup_isaacteleop_env.sh
#
# Creates a separate conda env (Isaac Sim's python must not see it). conda-forge channel only, so no
# Anaconda ToS prompt. The root overlay of this container is nearly full, so the env lives on /workspace
# and ~/.cloudxr (runtime state, EULA flag, logs, cloudxr.env) is a symlink to /workspace/.cloudxr.
set -eo pipefail

ENV_PREFIX=${ISAACTELEOP_ENV:-/workspace/envs/isaacteleop}
ISAACTELEOP_VERSION=${ISAACTELEOP_VERSION:-1.0.193}
CXR_HOME=${CXR_HOME:-/workspace/.cloudxr}

export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/workspace/.cache/pip}
export TMPDIR=${TMPDIR:-/workspace/.tmp}
mkdir -p "${TMPDIR}" "${CXR_HOME}"

if [ -e /root/.cloudxr ] && [ ! -L /root/.cloudxr ]; then
    echo "[setup] /root/.cloudxr exists and is not a symlink; move it to ${CXR_HOME} first." >&2
    exit 1
fi
ln -sfn "${CXR_HOME}" /root/.cloudxr

unset PYTHONPATH
source /opt/conda/etc/profile.d/conda.sh
if [ ! -x "${ENV_PREFIX}/bin/python" ]; then
    conda create -y -p "${ENV_PREFIX}" -c conda-forge --override-channels python=3.11
fi
conda activate "${ENV_PREFIX}"
pip install "isaacteleop[cloudxr,retargeters]==${ISAACTELEOP_VERSION}" --extra-index-url https://pypi.nvidia.com
python -c "import isaacteleop; print('[setup] isaacteleop', getattr(isaacteleop, '__version__', '?'), 'OK')"
echo "[setup] done: env ${ENV_PREFIX}, CloudXR state ${CXR_HOME} (-> /root/.cloudxr)"
