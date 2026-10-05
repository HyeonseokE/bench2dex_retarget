"""Download what the retargeting reads: teleop episodes and the assets of the selected tasks.

Runs inside the Isaac Sim container (huggingface_hub and yaml come from the setup step).

  episodes  Bench2Dex/teleopdata dataset/<task>/origin-generalization/episode_000000..N-1.hdf5
            (origin = trajectories without images, ~4 MB each). The coupling fit needs a few
            episodes of every robot's own tasks, which the first episodes of each task cover.
  assets    the whole Bench2Dex/Assets repo (~19 GB, once). Retargeting alone needs only the five
            UR5 robots and the task objects, but stage 5 renders with --restore-generalization,
            which rebuilds the recorded backdrop, table texture and clutter distractors.

Environment: TASKS ("06 12 ..."), EPISODES (count), B2D_ROOT, HF_TOKEN (anonymous downloads get
rate-limited by the Hub).
"""

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(os.environ.get("B2D_ROOT", "/workspace"))
TASKS = os.environ["TASKS"].split()
EPISODES = int(os.environ.get("EPISODES", "50"))

scenes = {p.stem: p for p in (ROOT / "Bench2Dex" / "scenes").glob("*.yaml")}
names = []
for t in TASKS:
    hit = sorted(n for n in scenes if n.startswith(t))
    if not hit:
        sys.exit(f"FATAL: no Bench2Dex scene for task {t}")
    names.append(hit[0])

ep_pats = [f"dataset/{n}/origin-generalization/episode_{i:06d}.hdf5" for n in names for i in range(EPISODES)]
print(f"[episodes] {len(names)} tasks x {EPISODES} -> {ROOT / 'b2d_origin'}", flush=True)
snapshot_download("Bench2Dex/teleopdata", repo_type="dataset", allow_patterns=ep_pats,
                  local_dir=str(ROOT / "b2d_origin"), max_workers=8)

print(f"[assets] Bench2Dex/Assets (all) -> {ROOT / 'assets'}", flush=True)
snapshot_download("Bench2Dex/Assets", repo_type="dataset", local_dir=str(ROOT / "assets"), max_workers=8)

missing = [str(p) for n in names for i in range(EPISODES)
           if not (p := ROOT / "b2d_origin" / "dataset" / n / "origin-generalization" / f"episode_{i:06d}.hdf5").exists()]
if missing:
    print(f"[warn] {len(missing)} episodes not on the Hub, e.g. {missing[:3]}", flush=True)
print("FETCH OK", flush=True)
