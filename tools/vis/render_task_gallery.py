r"""Render reset-state screenshots of one Bench2Dex task scene (no HDF5 needed).

Builds the scene from ``scenes/<task>.yaml`` with the default robot, resets,
lets the objects settle for a few physics steps, then renders one PNG per
camera view.  Fork-only tool (FORK.md): upstream files are untouched.

Usage (one process per task; wrapper: render_task_gallery.sh)::

    python tools/vis/render_task_gallery.py --task scenes/03_wine_glass_plate_balance.yaml \
        --out-dir /workspace/experiments/amp7/bench2dex/<run_id>/artifacts --headless

Views (``--views``): name=Tx,Ty,Tz,Rx,Ry,Rz (m, deg; OpenGL camera, extrinsic XYZ Euler).
Defaults: front (across the table, facing the robot), back (over the robot's
shoulder), top (straight down).
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from isaaclab.app import AppLauncher  # noqa: E402
from utils.isaac_rendering import configure_headless_camera_parity_experience  # noqa: E402

DEFAULT_VIEWS = [
    "front=0.0,1.0,2.0,50.0,0.0,180.0",
    "back=0.0,-1.36,2.48,50.0,0.0,0.0",
    "top=0.0,0.0,2.3,0.0,0.0,0.0",
]

parser = argparse.ArgumentParser(description="Render reset-state PNGs of a Bench2Dex task scene.")
parser.add_argument("--task", type=str, required=True, help="Scene YAML path.")
parser.add_argument("--out-dir", type=str, required=True, help="Directory for <task>_<view>.png.")
parser.add_argument("--views", type=str, nargs="*", default=DEFAULT_VIEWS, help="name=Tx,Ty,Tz,Rx,Ry,Rz")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--focal-length", type=float, default=10.5)
parser.add_argument("--horizontal-aperture", type=float, default=20.955)
parser.add_argument("--settle-steps", type=int, default=60, help="Physics steps before rendering.")
parser.add_argument("--warmup-renders", type=int, default=5)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
configure_headless_camera_parity_experience(args_cli, log_prefix="task_gallery")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors.camera import Camera, CameraCfg  # noqa: E402
from isaaclab.utils.io import load_yaml  # noqa: E402

from build import build_scene, resolve_table_spec  # noqa: E402
from robots import DEFAULT_ROBOT_KEY  # noqa: E402
from utils.runtime_helpers import (  # noqa: E402
    classify_interactive_objects,
    initialize_articulation_state,
    write_articulation_targets,
)


def _rpy_deg_to_quat_wxyz(rx: float, ry: float, rz: float) -> np.ndarray:
    """Extrinsic XYZ Euler (deg) -> quaternion (w, x, y, z)."""
    from scipy.spatial.transform import Rotation

    q_xyzw = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def _parse_views(items: list[str]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    views = []
    for item in items:
        name, _, pose = item.partition("=")
        vals = [float(v) for v in pose.split(",")]
        if len(vals) != 6:
            raise ValueError(f"view '{item}' must be name=Tx,Ty,Tz,Rx,Ry,Rz")
        views.append((name, np.array(vals[:3], dtype=np.float64), _rpy_deg_to_quat_wxyz(*vals[3:])))
    return views


def _create_camera(width: int, height: int, focal_length: float, horizontal_aperture: float) -> Camera:
    import omni.usd

    base_path = "/World/GalleryCamera"
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(base_path).IsValid():
        stage.RemovePrim(base_path)
    stage.DefinePrim(base_path, "Xform")
    cam = Camera(
        cfg=CameraCfg(
            prim_path=f"{base_path}/Sensor",
            update_period=0,
            width=width,
            height=height,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length,
                horizontal_aperture=horizontal_aperture,
                clipping_range=(0.01, 100.0),
            ),
        )
    )
    if not hasattr(cam, "_rep_registry"):
        cam._rep_registry = {}
    return cam


def _init_camera_after_reset(cam: Camera) -> None:
    initialize_impl = getattr(cam, "_initialize_impl", None)
    if callable(initialize_impl):
        initialize_impl()
    if hasattr(cam, "_is_initialized"):
        cam._is_initialized = True
    if not hasattr(cam, "_rep_registry"):
        cam._rep_registry = {}
    if not hasattr(cam, "_ALL_INDICES"):
        view = getattr(cam, "_view", None)
        count = int(getattr(view, "count", 1) or 1)
        cam._ALL_INDICES = torch.arange(count, device=getattr(cam, "_device", "cpu"), dtype=torch.long)


def _capture_rgb(cam: Camera) -> np.ndarray:
    cam.update(dt=0.0, force_recompute=True)
    rgb = cam.data.output.get("rgb")
    if rgb is None or rgb.numel() == 0:
        raise RuntimeError("camera returned no RGB data")
    rgb = rgb[0]
    if rgb.shape[-1] > 3:
        rgb = rgb[..., :3]
    return rgb.to(dtype=torch.uint8).contiguous().cpu().numpy()


def main() -> None:
    t0 = time.time()
    task_path = os.path.abspath(args_cli.task)
    task_name = os.path.splitext(os.path.basename(task_path))[0]
    out_dir = os.path.abspath(args_cli.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    views = _parse_views(args_cli.views)

    task = load_yaml(task_path)
    task_dir = os.path.dirname(task_path)
    _, table_h, _ = resolve_table_spec(task)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device=getattr(args_cli, "device", "cuda:0")))
    sim.set_camera_view([0.0, 1.0, table_h + 0.45], [0.0, 0.0, table_h])
    print(f"[task_gallery] {task_name}: building scene (robot={DEFAULT_ROBOT_KEY})", flush=True)

    runtime = build_scene(task, task_dir, robot_key=DEFAULT_ROBOT_KEY)
    interactive_objects = runtime["interactive_objects"]
    cam = _create_camera(args_cli.width, args_cli.height, args_cli.focal_length, args_cli.horizontal_aperture)

    sim.reset()
    groups = classify_interactive_objects(interactive_objects)
    hold_targets: dict[int, object] = {}
    for art in groups.controlled_articulations:
        hold_targets[id(art)] = initialize_articulation_state(art)
    for art in groups.task_articulations:
        initialize_articulation_state(art, set_hold_target=False)
    for obj in groups.other_objects:
        obj.reset()
    _init_camera_after_reset(cam)

    for _ in range(max(0, args_cli.settle_steps)):
        for hook in list(runtime.get("robot_runtime", {}).get("pre_step_hooks", [])):
            hook()
        write_articulation_targets(groups.controlled_articulations, hold_targets)
        sim.step(render=False)
    for _ in range(args_cli.warmup_renders):
        sim.render()

    saved = {}
    from PIL import Image

    for name, pos, quat in views:
        cam.set_world_poses(
            positions=torch.tensor(pos[None, :], dtype=torch.float32, device=sim.device),
            orientations=torch.tensor(quat[None, :], dtype=torch.float32, device=sim.device),
            convention="opengl",
        )
        for _ in range(3):
            sim.render()
        rgb = _capture_rgb(cam)
        png = os.path.join(out_dir, f"{task_name}_{name}.png")
        Image.fromarray(rgb).save(png)
        saved[name] = os.path.basename(png)
        print(f"[task_gallery] saved {png} {rgb.shape}", flush=True)

    meta = {
        "task": task_name,
        "description": task.get("description"),
        "robot_key": DEFAULT_ROBOT_KEY,
        "table_height": table_h,
        "objects": [o.get("id") for o in task.get("objects", []) if isinstance(o, dict)],
        "views": saved,
        "settle_steps": args_cli.settle_steps,
        "render_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, f"{task_name}.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    try:
        cam._rep_registry = {}
    except Exception:
        pass
    print(f"[task_gallery] done {task_name} in {time.time() - t0:.0f}s", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
