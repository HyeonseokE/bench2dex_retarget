"""Turn the states of a successful retargeted execution into a Bench2Dex episode.

A dataset episode is the record of what actually happened, not something re-simulated: SPIDER's
committed execution restores a snapshot at every commit, so its joint targets do not reproduce the
run when executed again open-loop (task 06 ep3: the apple is dropped at ~frame 200 on replay, in the
prototype's own env too, with 1 or 1024 envs). Stage 3 therefore keeps the state of env 0 after every
physics step of the committed execution (``capture``); this module writes those states the way
Bench2Dex's DataCollector writes a teleop episode, with Bench2Dex's own code, so the result goes
through its replay (kinematic: states are set, not simulated), label and export tools unchanged:

  state      robot {joint_names, qpos, qvel, qeffort} and object {pose_world xyzw, lin/ang velocity,
             articulation qpos/qvel/qeffort} -- the collector.state_reader layout
  frames     one per control step (step_stride = 3 physics steps), Convention A: frame t holds the
             observation and the joint target executed next; the first frame has none ("none")
  labels     collector.box_labeler.compute_box3d_labels from meta/local_bboxes, every frame
  metrics    benchmark.metric_tracker.MetricTracker fed every physics-step state, metrics_trace on
             (the released episodes carry metrics/timeseries); meta/success = its stable success
  writer     collector.hdf5_writer.HDF5EpisodeWriter ->
             <root>/<robot>/<scene>/origin-generalization/episode_<source episode>.hdf5

Meta is the source episode's (scene file, generalization sample, seeds, instruction, local bboxes,
...) so ``replay.py --restore-generalization`` rebuilds the same scene; robot_key, action names,
success, metrics and collection fields are the retargeted run's. Distractors were not simulated:
their recorded poses at the same frame index are written through.
"""

from __future__ import annotations

import copy
import fcntl
import gzip
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from . import paths


def _decode(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray):
        return _decode(v[()]) if v.shape == () else [_decode(x) for x in v.tolist()]
    if isinstance(v, np.generic):
        return v.item()
    return v


# ------------------------------------------------------------------ capture (inside Isaac)
def capture(env, i: int = 0) -> dict:
    """State of env i after a physics step, as plain numpy in the collector's layout."""
    objects = env.success_states(i)
    for k in env.art_ids:
        objects[k]["qeffort"] = env.objects[k].data.applied_torque[i].cpu().numpy().astype(np.float32)
    d = env.robot.data
    return {"robot": {"qpos": d.joint_pos[i].cpu().numpy().astype(np.float32),
                      "qvel": d.joint_vel[i].cpu().numpy().astype(np.float32),
                      "qeffort": d.applied_torque[i].cpu().numpy().astype(np.float32)},
            "objects": objects}


def trace_header(env) -> dict:
    """Everything the writer needs about the run that is not per-frame state."""
    d = env.robot.data
    soft = d.soft_joint_pos_limits[0].cpu().numpy()
    hard = d.joint_pos_limits[0].cpu().numpy() if getattr(d, "joint_pos_limits", None) is not None else None
    return {"robot_key": env.spec.key, "robot": env.spec.name, "joint_names": list(env.robot.joint_names),
            "obj_ids": list(env.obj_ids), "scene": env.scene_def.name, "table_z": float(env.table_heights.table_z),
            "physics_dt": float(env.physics_dt), "decimation": int(env.cfg.decimation),
            "joint_limits": {"soft": (soft[:, 0], soft[:, 1]), "hard": None if hard is None else (hard[:, 0], hard[:, 1]),
                             "soft_source": "runtime.soft_joint_pos_limits",
                             "hard_source": None if hard is None else "runtime.joint_pos_limits"}}


def save_trace(path: Path, header: dict, frames: list[dict]):
    """frames[t] = {"state": capture, "substeps": [capture, ...] (physics steps that led to t),
    "cmd": joint target executed from t on, or None}."""
    with gzip.open(path, "wb") as f:
        pickle.dump({"header": header, "frames": frames}, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_trace(path: Path) -> dict:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------ write (pure python)
def write_episode(trace: dict, task: str, episode: int, root: Path, extra_meta: dict,
                  stage_dir: str = "origin-generalization", only_success: bool = True):
    """Returns (path or None, stable_success, metrics episode payload or None)."""
    paths.use_bench2dex()
    from benchmark.metric_tracker import MetricTracker
    from collector.box_labeler import compute_box3d_labels
    from collector.episode_buffer import EpisodeBuffer, FrameRecord
    from collector.hdf5_writer import HDF5EpisodeWriter
    from collector.metrics_payload import build_metrics_episode_payload

    from .task import load_scene

    hdr, frames = trace["header"], trace["frames"]
    src = paths.episode_path(task, episode)
    with h5py.File(src, "r") as h:
        src_meta = {k: _decode(h[f"meta/{k}"][()]) for k in h["meta"]}
        src_step, src_ts = h["time/sim_step"][:], h["time/timestamp_ns"][:]
        distractors = {k: h[f"objects/{k}/pose_world"][:] for k in h["objects"] if k not in hdr["obj_ids"]}
    bb = json.loads(src_meta["local_bboxes"]) if isinstance(src_meta.get("local_bboxes"), str) else {}
    local_bboxes = {k: (np.asarray(v[0]), np.asarray(v[1])) for k, v in bb.items()}
    object_ids = list(hdr["obj_ids"]) + list(distractors)
    names = hdr["joint_names"]
    dt_ns = int(round(1e9 * hdr["physics_dt"]))

    def objects_at(state, i):
        out = copy.deepcopy(state["objects"])
        j = min(i, len(src_step) - 1)
        z = np.zeros(3, np.float32)
        for k, traj in distractors.items():
            out[k] = {"pose_world": traj[j].astype(np.float32), "lin_vel_world": z, "ang_vel_world": z}
        return out

    def robot_at(state):
        return {"joint_names": names, **state["robot"]}

    # MetricTracker, exactly as DataCollector builds and feeds it
    spec = copy.deepcopy(load_scene(hdr["scene"]).spec.get("metrics") or {})
    tracker = None
    if spec:
        spec.setdefault("safety", {})["table_z"] = hdr["table_z"]      # _metrics_spec_for_episode
        if isinstance(spec.get("grasp"), dict):
            spec["grasp"]["table_z"] = hdr["table_z"]
        spec["metrics_trace"] = True
        tracker = MetricTracker(spec, dt=hdr["physics_dt"], robot_key=hdr["robot_key"])
    sim_step, updates = int(src_step[0]), 0
    buf = EpisodeBuffer(camera_ids=[], object_ids=object_ids, max_frames=0)
    for i, fr in enumerate(frames):
        subs = fr.get("substeps") or []
        for st in subs:                                           # DataCollector.after_step
            sim_step += 1
            if tracker is not None and tracker.available:
                tracker.update(objects_at(st, i), sim_step=sim_step, dt=hdr["physics_dt"],
                               robot_state=robot_at(st), joint_limits=hdr["joint_limits"])
                updates += 1
        if not subs and i > 0:                                    # coarse trace (no physics substeps)
            sim_step += hdr["decimation"]
            if tracker is not None and tracker.available:
                tracker.update(objects_at(fr["state"], i), sim_step=sim_step,
                               dt=hdr["physics_dt"] * hdr["decimation"],
                               robot_state=robot_at(fr["state"]), joint_limits=hdr["joint_limits"])
                updates += 1
        objs = objects_at(fr["state"], i)
        cmd = fr.get("cmd") if i > 0 else None                    # first frame: no action, as collected
        buf.add_frame(FrameRecord(
            frame_index=i, timestamp_ns=int(src_ts[0]) + (sim_step - int(src_step[0])) * dt_ns, sim_step=sim_step,
            valid=True, errors=[], camera={}, robot=robot_at(fr["state"]), objects=objs,
            labels={"box3d": compute_box3d_labels(objs, local_bboxes)},
            action_commanded=None if cmd is None else np.asarray(cmd, np.float32), action_valid=cmd is not None,
            action_source="spider" if cmd is not None else "none"))
    result = tracker.finalize(steps=updates, terminated_reason="max_steps") if tracker and tracker.available else None
    payload = build_metrics_episode_payload(result) if result is not None else None
    success = bool(result.stable_success) if result is not None else False
    if only_success and not success:
        return None, success, payload

    meta = {k: v for k, v in src_meta.items() if k not in ("frame_count", "created_at")}
    sample = json.loads(meta["scene_generalization_sample"])
    sample["robot_key"] = hdr["robot_key"]
    if isinstance(sample.get("embodiment"), dict):
        sample["embodiment"]["robot_key"] = hdr["robot_key"]
    meta.update({
        "robot_key": hdr["robot_key"], "action_names": names, "action_dim": len(names),
        "scene_generalization_sample": json.dumps(sample), "success": success, "success_available": result is not None,
        "metrics_episode": payload, "metrics_timeseries": result.timeseries if result is not None else None,
        "homing_start_sim_step": -1, "collection_mode": "spider_retarget", "source_domain": "sim",
        "demo_eligible": success, "has_rgb": False, "frame_error_count": 0,
        "created_at": datetime.now(timezone.utc).isoformat(), **extra_meta,
    })
    writer = HDF5EpisodeWriter(str(root), hdr["robot"], hdr["scene"], stage_dir)
    with open(Path(writer.output_dir) / ".write.lock", "w") as lf:     # array tasks share the manifest
        fcntl.flock(lf, fcntl.LOCK_EX)
        out = writer.write_episode(buf, episode, meta)
    return Path(out), success, payload
