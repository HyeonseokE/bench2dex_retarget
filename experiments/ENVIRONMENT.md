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

Filled from `ondemand/sysinfo.sbatch` (writes `SPEC.md`: host, container, render check). `?` = not measured yet.

| item | value | source |
|---|---|---|
| node / GPU | p1, NVIDIA RTX PRO 6000 Blackwell 96 GB (sm_120) | sim check job, 2026-10-05 |
| NVIDIA driver / CUDA (driver) | 580.82.07 on 2026-10-05; **re-check** (suspected newer now) | sysinfo |
| OS / kernel / CPU / RAM | ? | sysinfo |
| SLURM / apptainer | ? / 1.5.2 | sysinfo |
| container | `nvcr.io/nvidia/isaac-sim:5.1.0` SIF, `--nv`, headless Vulkan via libEGL_nvidia ICD | ondemand/env.sh |
| torch / Isaac Lab (in container) | 2.7.0+cu128 (required for sm_120) / v2.3.2 | ondemand/setup.sh |
| disks | `$HOME` NFS (26 TB free); node `/` 98% full, keep caches off `/tmp` | 2026-10-05 |
| Isaac Sim render (Kit, camera) | ? | sysinfo R1, R2 |

Driver vs Isaac Sim 5.1: NVIDIA tested 5.1.0 on Linux driver **580.65.06** (R580). R590/R595 drivers
(CUDA 13.1/13.2) have reports of RTX-renderer crashes (`librtx.scenedb.plugin.so` segfault, DEVICE_LOST),
including on RTX PRO 6000 Blackwell. A container cannot change the host driver.

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
