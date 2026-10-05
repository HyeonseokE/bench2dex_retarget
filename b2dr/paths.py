"""Filesystem layout, identical on the dev box and inside the cluster container.

Everything hangs off one root, ``B2D_ROOT`` (default ``/workspace``). On the cluster the job binds
``$HOME/b2d`` there, so the same paths work in both places:

    $B2D_ROOT/Bench2Dex            benchmark code (scene yamls, success evaluators, robot spawners)
    $B2D_ROOT/dex2bench_dataset    -> assets (robots, objects); Bench2Dex's relative paths expect this name
    $B2D_ROOT/b2d_origin/dataset   teleop episodes (HF Bench2Dex/teleopdata, origin-generalization)
    $B2D_ROOT/b2dr_runs            everything this repo writes
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("B2D_ROOT", "/workspace"))
BENCH2DEX = Path(os.environ.get("B2D_BENCH2DEX", ROOT / "Bench2Dex"))
DATASET = Path(os.environ.get("B2D_DATASET", ROOT / "b2d_origin" / "dataset"))
RUNS = Path(os.environ.get("B2DR_RUNS", ROOT / "b2dr_runs"))


def use_bench2dex() -> None:
    """Make Bench2Dex importable and make its relative asset paths resolve.

    Its robot spawners open ``../dex2bench_dataset/...`` relative to the working directory and its
    scene yamls ``../../dex2bench_dataset/...`` relative to ``scenes/``, so the process must run from
    the Bench2Dex root.
    """
    if str(BENCH2DEX) not in sys.path:
        sys.path.insert(0, str(BENCH2DEX))
    os.chdir(BENCH2DEX)


def task_dir(task: str) -> Path:
    """Episode folder of a task given by number ('06') or full name."""
    hits = sorted(DATASET.glob(f"{task}*/origin-generalization"))
    if not hits:
        raise FileNotFoundError(f"no episodes for task {task!r} under {DATASET}")
    return hits[0]


def episode_path(task: str, episode: int) -> Path:
    p = task_dir(task) / f"episode_{episode:06d}.hdf5"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def run_dir(task: str, episode: int) -> Path:
    """Per-episode output folder: reference (stage 1) plus one sub-folder per target robot."""
    d = RUNS / task_dir(task).parent.name / f"ep{episode:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d
