"""Record a retargeted rollout exactly the way Bench2Dex records a teleop episode.

Everything that defines the format is Bench2Dex's own code, driven the way its DataCollector drives
it, so the output is an ``origin``-stage episode indistinguishable in layout from a collected one
and goes through the benchmark's replay (RGB + TacMap tactile), label and export tools unchanged:

  state      collector.state_reader.read_joint_state / read_joint_limits for the robot; object
             states in the same dict layout (pose_world xyzw, velocities, articulation qpos/qvel/
             qeffort). Env 0 is the committed rollout; its env origin is subtracted.
  frames     one per control step (step_stride = 3 physics steps), Convention A: frame t holds the
             observation and the joint target executed next; the first frame has none ("none").
  labels     collector.box_labeler.compute_box3d_labels from meta/local_bboxes, every frame.
  metrics    benchmark.metric_tracker.MetricTracker updated after every physics step, finalised
             into metrics/episode + metrics/timeseries; meta/success is its stable_success, the
             benchmark's official success.
  writer     collector.hdf5_writer.HDF5EpisodeWriter -> <root>/<robot>/<scene>/origin-generalization/
             episode_<source episode>.hdf5, the teleopdata layout.

Driven by scripts/stage4_record.py, which re-executes a SPIDER plan open-loop in a single env --
the same "command -> physics" execution a teleop recording is.

Meta is the source episode's (scene file, generalization sample, seeds, instruction, local
bboxes, ...) so ``replay.py --restore-generalization`` rebuilds the same scene; robot_key, action
names, success, metrics and the collection fields are the rollout's. Distractors are not simulated:
their recorded poses at the same frame index are written through.
"""

from __future__ import annotations

import copy
import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from . import paths


def _decode(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray):
        if v.shape == ():
            return _decode(v[()])
        return [_decode(x) for x in v.tolist()]
    if isinstance(v, np.generic):
        return v.item()
    return v


class Bench2DexRecorder:
    def __init__(self, env, source_path: Path):
        paths.use_bench2dex()
        from benchmark.metric_tracker import MetricTracker
        from collector.box_labeler import compute_box3d_labels
        from collector.state_reader import read_joint_limits, read_joint_state

        self._read_joint_state, self._read_joint_limits = read_joint_state, read_joint_limits
        self._box3d = compute_box3d_labels
        self.env = env
        with h5py.File(source_path, "r") as h:
            self.src_meta = {k: _decode(h[f"meta/{k}"][()]) for k in h["meta"]}
            self.src_sim_step = h["time/sim_step"][:]
            self.src_ts = h["time/timestamp_ns"][:]
            self.distractors = {k: h[f"objects/{k}/pose_world"][:] for k in h["objects"] if k not in env.obj_ids}
        self.object_ids = list(env.obj_ids) + list(self.distractors)
        bb = json.loads(self.src_meta["local_bboxes"]) if isinstance(self.src_meta.get("local_bboxes"), str) else {}
        self.local_bboxes = {k: (np.asarray(v[0]), np.asarray(v[1])) for k, v in bb.items()}
        self.stride = int(self.src_meta.get("step_stride", env.cfg.decimation))
        self.dt_ns = int(round(1e9 * env.physics_dt))

        spec = copy.deepcopy(env.scene_def.spec.get("metrics") or {})
        for key in ("safety", "grasp"):               # DataCollector._metrics_spec_for_episode
            if isinstance(spec.get(key), dict) or key == "safety":
                spec.setdefault(key, {})["table_z"] = float(env.table_heights.table_z)
        # The released episodes carry metrics/timeseries, which MetricTracker only builds with
        # metrics_trace on; the current collector never sets it (it is off since collection).
        if spec:
            spec["metrics_trace"] = True
        self.metrics = MetricTracker(spec, dt=env.physics_dt, robot_key=env.spec.key) if spec else None
        self.frames: list[dict] = []
        self.sim_step = int(self.src_sim_step[0])
        self.metric_updates = 0

    # ---------------------------------------------------------------- states
    def object_states(self, frame_index: int) -> dict:
        env = self.env
        states = env.success_states(0)
        for k in env.art_ids:
            states[k]["qeffort"] = env.objects[k].data.applied_torque[0].cpu().numpy().astype(np.float32)
        i = min(frame_index, len(self.src_sim_step) - 1)
        zero = np.zeros(3, np.float32)
        for k, traj in self.distractors.items():
            states[k] = {"pose_world": traj[i].astype(np.float32), "lin_vel_world": zero, "ang_vel_world": zero}
        return states

    def observe(self):
        """Capture frame len(frames) (call after the control step that produced it)."""
        i = len(self.frames)
        objects = self.object_states(i)
        self.frames.append({
            "sim_step": self.sim_step,
            "timestamp_ns": int(self.src_ts[0]) + (self.sim_step - int(self.src_sim_step[0])) * self.dt_ns,
            "robot": self._read_joint_state(self.env.robot),
            "objects": objects,
            "labels": {"box3d": self._box3d({k: v for k, v in objects.items()}, self.local_bboxes)},
            "cmd": None, "source": "none",
        })

    def set_action(self, cmd):
        """The joint target that is executed from the latest frame on (Convention A)."""
        if self.frames and len(self.frames) > 1:      # the first frame keeps no action, as collected
            self.frames[-1]["cmd"] = np.asarray(cmd, np.float32)
            self.frames[-1]["source"] = "spider"

    def after_physics_step(self):
        """DataCollector.after_step: one metric update per physics step."""
        self.sim_step += 1
        if self.metrics is None or not self.metrics.available:
            return
        self.metrics.update(self.object_states(len(self.frames)), sim_step=self.sim_step, dt=self.env.physics_dt,
                            robot_state=self._read_joint_state(self.env.robot),
                            joint_limits=self._read_joint_limits(self.env.robot))
        self.metric_updates += 1

    # ----------------------------------------------------------- backtrack
    def state(self):
        return (len(self.frames), copy.deepcopy(self.metrics), self.metric_updates, self.sim_step)

    def load(self, st):
        n, metrics, self.metric_updates, self.sim_step = st
        self.metrics = copy.deepcopy(metrics)
        del self.frames[n:]

    # --------------------------------------------------------------- write
    def finalize(self):
        """MetricTracker result + Bench2Dex payload, computed once (finalize is not idempotent)."""
        if getattr(self, "_fin", None) is None:
            if self.metrics is None or not self.metrics.available:
                return None
            from collector.metrics_payload import build_metrics_episode_payload
            result = self.metrics.finalize(steps=self.metric_updates, terminated_reason="max_steps")
            self._fin = (result, build_metrics_episode_payload(result))
        return self._fin

    def write(self, root: Path, episode_index: int, extra_meta: dict,
              stage_dir: str = "origin-generalization") -> tuple[Path, bool]:
        from collector.episode_buffer import EpisodeBuffer, FrameRecord
        from collector.hdf5_writer import HDF5EpisodeWriter

        fin = self.finalize()
        success = bool(fin[0].stable_success) if fin else False
        buf = EpisodeBuffer(camera_ids=[], object_ids=self.object_ids, max_frames=0)
        for i, fr in enumerate(self.frames):
            buf.add_frame(FrameRecord(
                frame_index=i, timestamp_ns=fr["timestamp_ns"], sim_step=fr["sim_step"], valid=True, errors=[],
                camera={}, robot=fr["robot"], objects=fr["objects"], labels=fr["labels"],
                action_commanded=fr["cmd"], action_valid=fr["cmd"] is not None, action_source=fr["source"]))
        meta = {k: v for k, v in self.src_meta.items() if k not in ("frame_count", "created_at")}
        names = list(self.env.robot.joint_names)
        sample = json.loads(meta["scene_generalization_sample"])
        sample["robot_key"] = self.env.spec.key
        if isinstance(sample.get("embodiment"), dict):
            sample["embodiment"]["robot_key"] = self.env.spec.key
        meta.update({
            "robot_key": self.env.spec.key, "action_names": names, "action_dim": len(names),
            "scene_generalization_sample": json.dumps(sample), "success": success, "success_available": fin is not None,
            "metrics_episode": fin[1] if fin else None, "metrics_timeseries": fin[0].timeseries if fin else None,
            "homing_start_sim_step": -1, "collection_mode": "spider_retarget", "source_domain": "sim",
            "demo_eligible": success, "has_rgb": False, "frame_error_count": 0,
            "created_at": datetime.now(timezone.utc).isoformat(), **extra_meta,
        })
        scene = self.env.scene_def.name
        writer = HDF5EpisodeWriter(str(root), self.env.spec.name, scene, stage_dir)
        lock = Path(writer.output_dir) / ".write.lock"
        with open(lock, "w") as lf:                 # array tasks share the manifest of a (robot, task)
            fcntl.flock(lf, fcntl.LOCK_EX)
            out = writer.write_episode(buf, episode_index, meta)
        return Path(out), success
