# Environment

## Dev box (where the 06 -> Shadow prototype and the pipeline tests ran)

| item | value |
|---|---|
| GPU | 1x NVIDIA RTX 3090 24 GB, driver 580.95.05 (shared with other jobs) |
| OS | Linux 6.8 container, headless (Vulkan via libEGL_nvidia ICD) |
| Python env | conda, Python 3.11.16 (`/workspace/bench2dex_env.sh`) |
| Simulator | Isaac Sim 5.1.0 (pip), Isaac Lab v2.3.2 |
| Libraries | torch 2.7.0, numpy 1.26.4, h5py 3.16, trimesh 4.5.1 |
| Benchmark | this fork of Bench2Dex (upstream github.com/Bench2Dex/Bench2Dex) |

Plain GPU servers: `tools/retarget/local/setup.sh` builds the same stack (torch cu128).
Cluster (pro6000, SLURM): `ondemand/` (Isaac Sim 5.1 SIF + Isaac Lab v2.3.2).

## Cluster: pro6000 (SLURM, OpenOnDemand)

From `ondemand/sysinfo.sbatch` job 1534 (2026-10-07; `SPEC.md`: host, container, render check).

| item | value | source |
|---|---|---|
| node / GPU | p1: 4x NVIDIA RTX PRO 6000 Blackwell Server Edition 96 GB (sm_120), Gres gpu:pro6000:4 | sysinfo job 1534, 2026-10-07 |
| NVIDIA driver / CUDA (driver) | **580.82.07** (open kernel module, R580 branch) / 13.0; GSP firmware 580.82.07; compute mode Default, MIG off | sysinfo |
| OS / kernel | Ubuntu 24.04.4, 6.8.0-134-generic, glibc 2.39 | sysinfo |
| CPU / RAM | 2x AMD EPYC 9135 16-core (Zen 5), 64 threads, max 4.3 GHz / 503 GB (CfgTRES cpu=64, mem=500000M) | sysinfo |
| SLURM / apptainer | slurm-wlm 23.11.4 / 1.5.2, nvidia-container-cli 1.13.5 | sysinfo |
| disks | node `/` (holds `/tmp`) 7.0 TB, 99% used, 116 GB free; `/home` NAS 44 TB, 26 TB free | sysinfo |
| container | `nvcr.io/nvidia/isaac-sim:5.1.0` SIF (Ubuntu 24.04.2, Isaac Sim 5.1.0-rc.19), `--nv` binds the 580.82.07 GL/EGL/Vulkan/rtcore/optix libs | sysinfo |
| torch / Isaac Lab (in container) | 2.7.0+cu128, sm_120 kernels, Python 3.11.13 / isaaclab 0.54.2 (v2.3.2) | sysinfo |
| Isaac Sim render | R1 Kit headless + render steps: **PASS** (250 s). R2 Isaac Lab camera RGB frame: **timeout at 1200 s** (NGX/DLSS context errors in the log; cause not identified yet) | sysinfo |

Driver vs Isaac Sim 5.1: NVIDIA tested 5.1.0 on Linux driver **580.65.06** (R580). R590/R595 drivers
(CUDA 13.1/13.2) have reports of RTX-renderer crashes (`librtx.scenedb.plugin.so` segfault, DEVICE_LOST),
including on RTX PRO 6000 Blackwell. A container cannot change the host driver. p1 runs 580.82.07 = the tested R580 branch.

## Simulation settings

Physics is Bench2Dex's and is not modified for retargeting: robots are spawned with the Bench2Dex
spawners (actuator gains, home pose, drive types, hand friction, gravity compensation), objects with
Bench2Dex's scene helpers (per-episode table height, object max depenetration 5.0). 60 Hz physics,
20 Hz control (= recording rate). Success = Bench2Dex MetricTracker stable success (all task
conditions held for dwell_time 0.5 s), checked every step.

## Capacity notes

- One SPIDER process (1024 samples) uses ~5-6 GB of GPU memory; at most 3 per 24 GB GPU. A CUDA
  out-of-memory error corrupts PhysX state (all objects froze in one pose and passed the success
  check on 06 ep4); stage 3 now stops with an error when that happens.
- Prototype cost on the 3090: 45-250 min per episode (2 episodes in parallel).

## Robot placement (decided 2026-10-07)

Robots are placed exactly as Bench2Dex places them (`robots/multi_ur5_*_with_flange.py`): Shadow base at x = 0.75 m,
RH56DFX / RH5DG2 / Schunk / Wuji at x = 0.5 m. The Shadow teleop demos (43, 76) were collected there and policies are
evaluated there, so retargeted Shadow data keeps it; the arm IK absorbs the offset (fingertips are matched in world frame).
Moving bases to a common pose would be an environment change and is not done.
