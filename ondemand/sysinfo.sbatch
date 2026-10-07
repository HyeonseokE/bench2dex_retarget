#!/bin/bash
# Software spec of a pro6000 node + an Isaac Sim rendering check -- SINGLE ENTRY POINT.
#
#   OOD Job Composer -> New Job -> paste THIS FILE -> Submit.
#
# Purpose: the NVIDIA driver on the node decides whether Isaac Sim 5.1's RTX renderer works (tested
# driver 580.65.06; R590/R595 drivers have renderer crash reports). This job records what the node
# runs and whether rendering works, nothing else (no data download, no HF token needed).
#
#   [A] host       OS, kernel, CPU, RAM, GPU, driver / CUDA, kernel module, host graphics libs,
#                  SLURM, apptainer, disks
#   [B] container  Isaac Sim image: OS, Isaac Sim version, GPU libs bound by --nv, torch, Isaac Lab
#   [C] render     R1 Kit headless + render steps (checks/l3_isaacsim.py)
#                  R2 Isaac Lab camera, one RGB frame -> PNG (checks/l4_camera.py --enable_cameras)
#                  renderer lines of the Kit log (driver seen by Kit, Vulkan, rtx errors)
#
# Output: SPEC.md (also printed at the end) in $HOME/b2d/b2dr_runs_envcheck/sysinfo/<jobid>/ with
# the per-check logs and the PNG. The first run builds the SIF and the Python env (~1 h); later ~10 min.
# Knob: NVCCLI=1 binds GPU libs by capability (try it if R1/R2 fail with missing graphics libs).
#
#SBATCH --job-name=b2dr-sysinfo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-b2dr-sysinfo-%j.out

set -uo pipefail
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"
echo "=== b2dr-sysinfo start: $(date) on $(hostname) ==="
REPO_URL="${REPO_URL:-https://github.com/HyeonseokE/bench2dex_retarget.git}"

WS="$HOME/b2d"
B="$WS/bench2dex_retarget"
OUT="$WS/b2dr_runs_envcheck/sysinfo/${SLURM_JOB_ID:-manual-$$}"
mkdir -p "$OUT"
SPEC="$OUT/SPEC.md"
sec() { printf '\n## %s\n\n```\n' "$1" >> "$SPEC"; }
end() { printf '```\n' >> "$SPEC"; }
run() { echo "\$ $*" >> "$SPEC"; timeout 120 bash -c "$*" >> "$SPEC" 2>&1 || echo "(failed or not available)" >> "$SPEC"; }

printf '# pro6000 node spec\n\njob %s, node %s, %s\n' "${SLURM_JOB_ID:-manual}" "$(hostname)" "$(date '+%F %T')" > "$SPEC"

echo "=== [A] host ==="
sec "A1 OS / kernel / CPU / RAM"
run "grep -E '^(PRETTY_NAME|VERSION_ID)' /etc/os-release"
run "uname -r"
run "lscpu | grep -E 'Model name|^Socket|^Core|^Thread|^CPU\(s\)|MHz'"
run "echo allocated cpus: \${SLURM_CPUS_ON_NODE:-?}; nproc"
run "free -g | head -2"
end
sec "A2 GPU / driver / CUDA"
run "nvidia-smi --query-gpu=index,name,driver_version,vbios_version,memory.total,pcie.link.gen.max,pcie.link.width.max,persistence_mode,display_mode,display_active,compute_mode,mig.mode.current --format=csv"
run "nvidia-smi | head -4 | tail -2"
run "cat /proc/driver/nvidia/version"
run "nvidia-smi -q | grep -iE 'GSP Firmware|Virtualization Mode|Addressing Mode' | sort -u"
end
sec "A3 host NVIDIA graphics/compute libs (what --nv can bind)"
run "ls -1 /usr/lib/x86_64-linux-gnu/ /usr/lib64/ 2>/dev/null | grep -E '^lib(EGL_nvidia|GLX_nvidia|nvidia-(rtcore|gpucomp|glcore|eglcore|glsi|ml|vulkan-producer)|nvoptix|cuda)\.so\.[0-9]' | sort -u"
run "ls /usr/share/vulkan/icd.d/ /etc/vulkan/icd.d/ /usr/share/glvnd/egl_vendor.d/ 2>/dev/null"
end
sec "A4 SLURM / apptainer / disks"
run "sinfo --version"
run "scontrol show node \$(hostname -s) | grep -E 'NodeName|Gres=|CfgTRES|OS='"
run "apptainer --version"
run "command -v nvidia-container-cli && nvidia-container-cli --version | head -2"
run "ldd --version | head -1"
run "df -h /tmp \$HOME | sed 's/^/  /'"
end

echo "=== git sync + setup ==="
export APPTAINER_CACHEDIR="/tmp/b2d-$USER/apptainer-cache"; mkdir -p "$APPTAINER_CACHEDIR" "$WS"
LG() { apptainer exec --bind "$WS" docker://hyeonseoke/lerobot:v1 git "$@"; }
if [ -d "$B/.git" ]; then LG -C "$B" pull --ff-only || echo "WARN: git pull failed; using the existing checkout"
else LG clone "$REPO_URL" "$B" || { echo "FATAL: clone failed"; exit 1; }; fi
# shellcheck source=/dev/null
source "$B/ondemand/env.sh"
# shellcheck source=/dev/null
( set -e; source "$CODE_DIR/setup.sh" ) || echo "WARN: setup did not finish; container checks may fail"

echo "=== [B] container ==="
sec "B Isaac Sim container"
run "${ISAAC[*]} bash -c 'grep PRETTY_NAME /etc/os-release; cat /isaac-sim/VERSION 2>/dev/null; echo; ls /.singularity.d/libs/ | grep -E \"nvidia|EGL|GLX|nvoptix|cuda\" | sort'"
run "${ISAAC[*]} $PY -c 'import sys,torch; print(\"python\", sys.version.split()[0]); print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"available\", torch.cuda.is_available()); print(\"archs\", torch.cuda.get_arch_list()); print(\"device\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'"
run "${ISAAC[*]} $PY -c 'import importlib.metadata as m; print(\"isaaclab\", m.version(\"isaaclab\"))'"
run "${ISAAC[*]} bash -c 'command -v vulkaninfo && vulkaninfo --summary 2>&1 | grep -E \"apiVersion|driverVersion|deviceName|driverName\" | head -8'"
end

echo "=== [C] render ==="
CK=/workspace/bench2dex_retarget/ondemand/checks
COUT="/workspace/b2dr_runs_envcheck/sysinfo/$(basename "$OUT")"
declare -A RES
check() {   # check NAME MARKER TIMEOUT cmd...
  local name="$1" marker="$2" tmo="$3"; shift 3
  local t0=$SECONDS rc=0
  timeout --kill-after=30 "$tmo" "$@" > "$OUT/$name.log" 2>&1 || rc=$?
  if grep -q "$marker" "$OUT/$name.log"; then RES[$name]="PASS"
  elif [ $rc -eq 124 ] || [ $rc -eq 137 ]; then RES[$name]="FAIL (timeout)"
  elif grep -qiE "segmentation fault|signal 11|crash" "$OUT/$name.log"; then RES[$name]="FAIL (crash, rc=$rc)"
  else RES[$name]="FAIL (rc=$rc)"; fi
  RES[$name]="${RES[$name]} in $(( SECONDS - t0 ))s"
  echo "--- $name: ${RES[$name]}"
}
check R1 L3_OK 1200 "${ISAAC[@]}" "$PY" $CK/l3_isaacsim.py
check R2 L4C_OK 1200 "${ISAAC[@]}" "$PY" $CK/l4_camera.py --headless --enable_cameras --num_envs 1 --png "$COUT/r2_camera.png"
sec "C render checks"
for k in R1 R2; do echo "$k: ${RES[$k]}" >> "$SPEC"; done
echo >> "$SPEC"
echo "renderer lines (R2 log):" >> "$SPEC"
grep -hiE "driver version|active gpu|gpu foundation|vulkan|rtx|librtx|device_lost|segmentation|\| *[0-9] *\|.*\|" "$OUT/R2.log" \
  | grep -viE "warn.*deprecated" | sort -u | head -40 | cut -c1-200 >> "$SPEC"
end

echo
echo "=============================== SPEC.md ($SPEC) ==============================="
cat "$SPEC"
echo "logs: $OUT   camera frame: $OUT/r2_camera.png"
