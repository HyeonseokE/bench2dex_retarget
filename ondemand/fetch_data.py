"""Download what the retargeting reads: teleop episodes and the assets of the selected tasks.

Runs inside the Isaac Sim container (huggingface_hub and yaml come from the setup step).

  episodes  Bench2Dex/teleopdata dataset/<task>/origin-generalization/episode_000000..N-1.hdf5 only: the *_1
            files are the same trajectories with another background/texture/lighting sample, and
            replay-generalization/ is their rendered version, so neither is needed
            (origin = trajectories without images, ~4 MB each), plus episodes 0-4 of one task per
            hand (06 RH56DFX, 07 RH5DG2, 43 Shadow, 08 Schunk, 21 Wuji): retarget.coupling fits each
            target hand's mimic joints on that hand's own demos, whatever task is being retargeted.
  assets    from Bench2Dex/Assets only the five UR5 robots, the objects of the 12 UR5 tasks and scenes/
            (~1.8 GB): retargeting does not render. Rendering (stage 5, --restore-generalization)
            would also need Background/, textures/ and the clutter objects.

Environment: TASKS ("06 12 ..."), EPISODES (count), B2D_ROOT, HF_TOKEN (anonymous downloads get
rate-limited by the Hub).
"""

import os
import re
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
    """snapshot_download that survives the Hub's rate limit (1000 API requests / 5 min per user; every
    file costs requests, and Assets has thousands): on HTTP 429 wait out the window and resume --
    files already on disk are skipped."""
    from huggingface_hub.errors import HfHubHTTPError
    stop = threading.Event()
    threading.Thread(target=progress, args=(dest, stop), daemon=True).start()
    try:
        for attempt in range(30):
            try:
                snapshot_download(repo, repo_type="dataset", local_dir=str(dest), max_workers=4, **kw)
                return
            except HfHubHTTPError as e:
                code = getattr(getattr(e, "response", None), "status_code", None)
                if code not in (429, 500, 502, 503, 504) or attempt == 29:
                    raise
                wait = 330 if code == 429 else 60
                print(f"  [{dest.name}] HTTP {code} ({'rate limit' if code == 429 else 'server error'}), "
                      f"resuming in {wait} s (retry {attempt + 1}/29)", flush=True)
                time.sleep(wait)
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

UR5_TASKS = ["06", "12", "42", "07", "34", "60", "43", "76", "08", "44", "21", "27"]
COUPLING_TASKS = ["06", "07", "43", "08", "21"]                # one source task per hand
coupling = [sorted(n for n in scenes if n.startswith(t))[0] for t in COUPLING_TASKS]
ep_pats = [f"dataset/{n}/origin-generalization/episode_{i:06d}.hdf5" for n in names for i in range(EPISODES)]
ep_pats += [f"dataset/{n}/origin-generalization/episode_{i:06d}.hdf5" for n in coupling for i in range(5)]
print(f"[episodes] {len(names)} tasks x {EPISODES} -> {ROOT / 'b2d_origin'}", flush=True)
fetch("Bench2Dex/teleopdata", ROOT / "b2d_origin", allow_patterns=ep_pats)

# Assets: only what retargeting (stages 1-4, no rendering) reads for the 12 UR5 tasks -- the five UR5 robots,
# the objects named in those scene files, and scenes/ -- ~1.8 GB instead of the whole 19 GB repo
# (backgrounds, textures, clutter objects and other robot sets are only needed to render).
objects = sorted({m for t in UR5_TASKS for n in scenes if n.startswith(t)
                  for m in re.findall(r"Objects/([^/\s'\"]+)/", scenes[n].read_text())})
asset_pats = ["assets.txt", "scenes/*", "Robots_p/ur5+*"] + [f"Objects/{o}/*" for o in objects]
print(f"[assets] Bench2Dex/Assets: 5 UR5 robots + {len(objects)} task objects + scenes -> {ROOT / 'assets'}", flush=True)
fetch("Bench2Dex/Assets", ROOT / "assets", allow_patterns=asset_pats)

missing = [str(p) for n in names for i in range(EPISODES)
           if not (p := ROOT / "b2d_origin" / "dataset" / n / "origin-generalization" / f"episode_{i:06d}.hdf5").exists()]
if missing:
    print(f"[warn] {len(missing)} episodes not on the Hub, e.g. {missing[:3]}", flush=True)
print("FETCH OK", flush=True)
