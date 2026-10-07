"""Download what the retargeting reads: teleop episodes and the assets of the selected tasks.

Runs inside the Isaac Sim container (huggingface_hub and yaml come from the setup step).

  episodes  Bench2Dex/teleopdata dataset/<task>/origin-generalization/episode_000000..N-1.hdf5 only: the *_1
            files are the same trajectories with another background/texture/lighting sample, and
            replay-generalization/ is their rendered version, so neither is needed
            (origin = trajectories without images, ~4 MB each), plus episodes 0-4 of one task per
            hand (06 RH56DFX, 07 RH5DG2, 43 Shadow, 08 Schunk, 21 Wuji): retarget.coupling fits each
            target hand's mimic joints on that hand's own demos, whatever task is being retargeted.
  assets    the whole Bench2Dex/Assets repo (~19 GB, once). Retargeting alone needs only the five
            UR5 robots and the task objects, but stage 5 renders with --restore-generalization,
            which rebuilds the recorded backdrop, table texture and clutter distractors.

Environment: TASKS ("06 12 ..."), EPISODES (count), B2D_ROOT, HF_TOKEN (anonymous downloads get
rate-limited by the Hub).
"""

import os
import sys
import threading
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(os.environ.get("B2D_ROOT", "/workspace"))
TASKS = os.environ["TASKS"].split()
EPISODES = int(os.environ.get("EPISODES", "50"))

scenes = {p.stem: p for p in (Path(__file__).resolve().parents[1] / "scenes").glob("*.yaml")}   # this repo


def progress(path: Path, stop: threading.Event, every: int = 60):
    """Progress bars are off in SLURM logs; print the bytes on disk under `path` every minute instead."""
    t0, last = time.time(), None
    while not stop.wait(every):
        n = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        rate = "" if last is None else f", {(n - last) / every / 2**20:.1f} MB/s"
        print(f"  [{path.name}] {n / 2**30:.2f} GB on disk after {(time.time() - t0) / 60:.0f} min{rate}", flush=True)
        last = n


def fetch(repo: str, dest: Path, **kw):
    stop = threading.Event()
    threading.Thread(target=progress, args=(dest, stop), daemon=True).start()
    try:
        snapshot_download(repo, repo_type="dataset", local_dir=str(dest), max_workers=8, **kw)
    finally:
        stop.set()


try:
    print(f"[hf] authenticated as {HfApi().whoami()['name']}", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"[hf] WARNING: not authenticated ({type(e).__name__}); downloads will be slow / rate-limited", flush=True)
names = []
for t in TASKS:
    hit = sorted(n for n in scenes if n.startswith(t))
    if not hit:
        sys.exit(f"FATAL: no Bench2Dex scene for task {t}")
    names.append(hit[0])

COUPLING_TASKS = ["06", "07", "43", "08", "21"]                # one source task per hand
coupling = [sorted(n for n in scenes if n.startswith(t))[0] for t in COUPLING_TASKS]
ep_pats = [f"dataset/{n}/origin-generalization/episode_{i:06d}.hdf5" for n in names for i in range(EPISODES)]
ep_pats += [f"dataset/{n}/origin-generalization/episode_{i:06d}.hdf5" for n in coupling for i in range(5)]
print(f"[episodes] {len(names)} tasks x {EPISODES} -> {ROOT / 'b2d_origin'}", flush=True)
fetch("Bench2Dex/teleopdata", ROOT / "b2d_origin", allow_patterns=ep_pats)

print(f"[assets] Bench2Dex/Assets (all) -> {ROOT / 'assets'}", flush=True)
fetch("Bench2Dex/Assets", ROOT / "assets")

missing = [str(p) for n in names for i in range(EPISODES)
           if not (p := ROOT / "b2d_origin" / "dataset" / n / "origin-generalization" / f"episode_{i:06d}.hdf5").exists()]
if missing:
    print(f"[warn] {len(missing)} episodes not on the Hub, e.g. {missing[:3]}", flush=True)
print("FETCH OK", flush=True)
