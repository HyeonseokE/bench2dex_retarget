"""A Bench2Dex task (scene yaml) and one of its teleop episodes (HDF5), in plain numpy.

Conventions used everywhere in b2dr: poses are (x, y, z, qw, qx, qy, qz) in the Bench2Dex world
frame (Isaac Lab's wxyz); the HDF5 stores xyzw and is converted on load. Arrays are truncated at
``meta/homing_start_sim_step`` -- Bench2Dex's own rule: the return-to-home motion is not part of
the demonstration.

Only task objects (``obj_*``) are retargeted. Clutter distractors are not simulated, as in the
task-06 prototype; the exporter copies their recorded poses through unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import yaml

from . import paths


def xyzw_to_wxyz(p):
    return np.concatenate([p[..., :3], p[..., 6:7], p[..., 3:6]], -1)


def wxyz_to_xyzw(p):
    return np.concatenate([p[..., :3], p[..., 4:7], p[..., 3:4]], -1)


@dataclass
class SceneObject:
    obj_id: str
    asset_key: str
    body_type: str              # dynamic | articulation
    usd_path: str
    scale: tuple | None
    asset_spec: dict
    obj_spec: dict


@dataclass
class Scene:
    name: str
    path: Path
    spec: dict
    objects: list[SceneObject]

    @property
    def success_conditions(self) -> list:
        return list(self.spec.get("success_conditions") or [])

    @property
    def dwell_s(self) -> float:
        """Bench2Dex's stable-success hold time (metrics.terminal.dwell_time_s, default 0.5 s)."""
        term = (self.spec.get("metrics") or {}).get("terminal") or {}
        return float(term.get("dwell_time_s", 0.5))


def load_scene(name: str) -> Scene:
    """``name``: scene file stem or the HDF5's meta/scene_file (an absolute path on the recorder)."""
    paths.use_bench2dex()
    from build.spawners import _resolve_path  # Bench2Dex

    stem = Path(str(name)).stem
    path = paths.BENCH2DEX / "scenes" / f"{stem}.yaml"
    spec = yaml.safe_load(open(path))
    objs = []
    for o in spec.get("objects", []):
        a = spec["assets"][o["asset"]]
        objs.append(SceneObject(
            obj_id=o["id"], asset_key=o["asset"], body_type=str(a.get("body_type", "dynamic")).lower(),
            usd_path=_resolve_path(a["path"], str(path.parent)),
            scale=tuple(a["scale"]) if "scale" in a else None, asset_spec=a, obj_spec=o))
    bad = [o.obj_id for o in objs if o.body_type not in ("dynamic", "articulation")]
    if bad:
        raise NotImplementedError(f"{stem}: body types other than dynamic/articulation: {bad}")
    return Scene(name=stem, path=path, spec=spec, objects=objs)


@dataclass
class Episode:
    path: Path
    index: int
    robot_key: str
    scene_name: str
    fps: int
    physics_dt: float
    step_stride: int
    n: int                                   # frames kept (before homing)
    joint_names: list[str]
    qpos: np.ndarray                         # (n, J) in joint_names order
    obj_pose: dict                           # obj_id -> (n, 7) wxyz
    obj_qpos: dict = field(default_factory=dict)        # articulation obj_id -> (n, nj)
    obj_joint_names: dict = field(default_factory=dict)
    sample: dict = field(default_factory=dict)          # scene_generalization_sample
    success: bool = False

    @property
    def dt(self) -> float:
        return 1.0 / self.fps

    def table_heights(self, scene: Scene):
        """Bench2Dex's TableHeights for this episode (generalization shifts the table, not the robot)."""
        paths.use_bench2dex()
        from build.table_geometry import resolve_table_geometry, resolve_table_heights  # no Isaac imports

        t = scene.spec.get("table") or {}            # build.scene_builder.resolve_table_spec (imports Isaac)
        size = tuple(float(v) for v in t.get("size", [2.2, 1.1, 0.04]))
        nominal_z = float(t.get("height", 0.75))
        h = resolve_table_heights(nominal_z, self.sample, generalization_enabled=True)
        return size, h, resolve_table_geometry(size, table_z=h.table_z)


def load_episode(task: str, index: int) -> Episode:
    path = paths.episode_path(task, index)
    with h5py.File(path, "r") as h:
        m = h["meta"]
        sim_step = h["time/sim_step"][:]
        homing = int(m["homing_start_sim_step"][()])
        n = int(np.searchsorted(sim_step, homing)) if homing > 0 else len(sim_step)
        names = [x.decode() for x in h["robot/joint_names"][:]]
        obj_pose, obj_qpos, obj_jn = {}, {}, {}
        for k in h["objects"]:
            if not k.startswith("obj_"):
                continue
            g = h[f"objects/{k}"]
            obj_pose[k] = xyzw_to_wxyz(g["pose_world"][:n].astype(np.float64))
            if "qpos" in g:
                obj_qpos[k] = g["qpos"][:n].astype(np.float64)
                obj_jn[k] = [x.decode() for x in g["joint_names"][:]]
        ep = Episode(
            path=path, index=index, robot_key=m["robot_key"][()].decode(),
            scene_name=Path(m["scene_file"][()].decode()).stem, fps=int(m["fps"][()]),
            physics_dt=float(m["physics_dt"][()]), step_stride=int(m["step_stride"][()]), n=n,
            joint_names=names, qpos=h["robot/qpos"][:n].astype(np.float64), obj_pose=obj_pose,
            obj_qpos=obj_qpos, obj_joint_names=obj_jn,
            sample=json.loads(m["scene_generalization_sample"][()]), success=bool(m["success"][()]))
    return ep


def list_episodes(task: str) -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in paths.task_dir(task).glob("episode_??????.hdf5"))
