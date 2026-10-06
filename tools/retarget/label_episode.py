"""GT labels for ONE rendered episode, with generate_gt_labels.py's defaults.

Bench2Dex's tools/labels/generate_gt_labels.py labels a whole directory (and rewrites every file in
it); array tasks render episodes of the same (robot, task) concurrently, so each labels its own file
with the same functions and settings: occupancy_gt (voxel 0.01 m, collision geometry, per-episode
auto bounds) and box3d + box2d, written in place.

  python tools/retarget/label_episode.py <replay-generalization/episode_XXXXXX.hdf5>
"""

import os
import sys

B2D = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Bench2Dex root
sys.path.insert(0, B2D)
os.chdir(B2D)
from collector.config import normalize_occupancy_gt_config  # noqa: E402
from tools.labels.generate_box_labels import generate_box_labels_for_file  # noqa: E402
from tools.labels.generate_occupancy_gt import generate_occupancy_gt_for_file  # noqa: E402

path = os.path.abspath(sys.argv[1])
cfg = normalize_occupancy_gt_config({"voxel_size": 0.01, "geometry_source": "collision",
                                     "label_version": "occupancy_gt_v1"})
ok_occ = generate_occupancy_gt_for_file(path, occupancy_config=cfg, write_mode="inplace", overwrite=True,
                                        device="cuda", bounds_override=None)
ok_box = generate_box_labels_for_file(path, box2d=True, write_mode="inplace", overwrite=True, device="cuda")
print(f"RESULT labels {path}: occupancy {bool(ok_occ)} boxes {bool(ok_box)}")
sys.exit(0 if ok_occ and ok_box else 1)
