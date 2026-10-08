#!/bin/bash
# Install and start Tailscale inside this container (no /dev/net/tun → userspace networking).
#
#   bash tools/teleop_xr/setup_tailscale.sh            # install (once) + start tailscaled + `tailscale up`
#   tools/teleop_xr/ts status                          # wrapper for the tailscale CLI afterwards
#
# `tailscale up` prints a login URL the first time; open it and approve the machine in your tailnet
# (the Quest must be logged into the same tailnet). State lives on /workspace, so it survives restarts
# of tailscaled; rerun this script after a container restart.
#
# Userspace mode: connections from tailnet peers to this node's 100.x address are delivered to local
# (127.0.0.1) listeners. If the headset signs in but the stream never starts, fall back to running
# Tailscale on the host and the container with --network host (tools/teleop_xr/README.md).
set -eo pipefail

TS_VERSION=${TS_VERSION:-1.80.3}
TS_DIR=${TS_DIR:-/workspace/envs/tailscale}
TS_HOSTNAME=${TS_HOSTNAME:-b2d-teleop}
SOCK="${TS_DIR}/tailscaled.sock"

if [ ! -x "${TS_DIR}/tailscaled" ]; then
    mkdir -p "${TS_DIR}"
    tmp=$(mktemp -d -p "${TMPDIR:-/workspace/.tmp}")
    url="https://pkgs.tailscale.com/stable/tailscale_${TS_VERSION}_amd64.tgz"
    echo "[tailscale] downloading ${url}"
    curl -fsSL "${url}" -o "${tmp}/ts.tgz"
    tar -xzf "${tmp}/ts.tgz" -C "${tmp}"
    cp "${tmp}/tailscale_${TS_VERSION}_amd64/tailscale" "${tmp}/tailscale_${TS_VERSION}_amd64/tailscaled" "${TS_DIR}/"
    rm -rf "${tmp}"
fi

if ! "${TS_DIR}/tailscale" --socket="${SOCK}" status >/dev/null 2>&1 && ! pgrep -x tailscaled >/dev/null; then
    echo "[tailscale] starting tailscaled (userspace networking)"
    nohup "${TS_DIR}/tailscaled" --tun=userspace-networking --socket="${SOCK}" \
        --statedir="${TS_DIR}/state" > "${TS_DIR}/tailscaled.log" 2>&1 &
    for _ in $(seq 1 50); do [ -S "${SOCK}" ] && break; sleep 0.2; done
fi

"${TS_DIR}/tailscale" --socket="${SOCK}" up --hostname="${TS_HOSTNAME}" "$@"
echo "[tailscale] this node: $("${TS_DIR}/tailscale" --socket="${SOCK}" ip -4 | head -n 1)"
