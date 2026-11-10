"""
xr_hand_source.py — OpenXR hand tracking (Meta Quest 3 via CloudXR) as a teleop input source.

Data path (same as DexVerse branch teleop/quest-cloudxr6):
  Quest 3 browser (WebXR, IsaacTeleop client) → CloudXR 6 runtime (isaacteleop)
  → Isaac Sim OpenXR (XR kit) → isaaclab.devices.openxr.OpenXRDevice
  → XrHandSource.poll(): per hand 26 joints [x, y, z, qw, qx, qy, qz] in the *sim world* frame
    (XrCfg anchor applied, so no extra axis mapping is needed).

Conversions used by the Bench2Dex retarget pipeline:
  * openxr_to_mediapipe(): 26 OpenXR joints → MediaPipe 21 keypoints. Drops palm and the
    index/middle/ring/little metacarpals (MediaPipe has no such points); keeps thumb_metacarpal
    (= MediaPipe thumb CMC). Same selection as DexVerse DEX_RETARGETING_HAND_JOINT_INDICES.
  * canonical_wrist_frame(): OpenXR wrist joint orientation → the wrist frame RetargetBridge
    expects (the frame _wrist_frame_from_quat builds from the Manus wrist node). With OpenXR
    joint axes (-Z along the bones toward the fingertips, +Y out of the back of the hand,
    X = Y × Z) it is (+Y, -X, +Z), i.e. the joint frame turned +90° about its Z axis, for both
    hands. Derived by aligning a synthetic open hand to the zero-pose fingertips of the UR5
    hands (all five share one root-frame convention: fingers +z, index side +y; residual < 6°)
    and checked by teleop/tests/test_xr_retarget.py.

NpzHandSource replays a recorded dump (B2D_XR_DUMP) through the same interface, so the
retarget / arm-IK path can be exercised without a headset.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from .math_utils import quat_wxyz_to_rotmat

# OpenXR XR_EXT_hand_tracking joint order (identical to isaaclab.devices.openxr.common.HAND_JOINT_NAMES;
# duplicated so this module imports without Isaac Sim for offline tests).
HAND_JOINT_NAMES = (
    "palm", "wrist",
    "thumb_metacarpal", "thumb_proximal", "thumb_distal", "thumb_tip",
    "index_metacarpal", "index_proximal", "index_intermediate", "index_distal", "index_tip",
    "middle_metacarpal", "middle_proximal", "middle_intermediate", "middle_distal", "middle_tip",
    "ring_metacarpal", "ring_proximal", "ring_intermediate", "ring_distal", "ring_tip",
    "little_metacarpal", "little_proximal", "little_intermediate", "little_distal", "little_tip",
)
NUM_XR_JOINTS = len(HAND_JOINT_NAMES)  # 26
WRIST = HAND_JOINT_NAMES.index("wrist")

# MediaPipe 21: wrist, thumb CMC/MCP/IP/TIP, then MCP/PIP/DIP/TIP for index..little.
OPENXR_TO_MEDIAPIPE = (1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25)

# Columns = canonical axes expressed in the OpenXR wrist joint frame: (+Y, -X, +Z).
OPENXR_JOINT_TO_CANONICAL = np.array(
    [[0.0, -1.0, 0.0],
     [1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0]],
    dtype=np.float32,
)

SIDES = ("right", "left")


def openxr_to_mediapipe(joints: np.ndarray) -> np.ndarray:
    """(26, 7) OpenXR joint poses → (21, 3) MediaPipe keypoint positions."""
    return np.asarray(joints, dtype=np.float32)[list(OPENXR_TO_MEDIAPIPE), :3]


def canonical_wrist_frame(wrist_quat_wxyz: np.ndarray) -> np.ndarray:
    """OpenXR wrist orientation (w, x, y, z) → (3, 3) RetargetBridge wrist frame (columns in world)."""
    R = quat_wxyz_to_rotmat(np.asarray(wrist_quat_wxyz, dtype=np.float64)).astype(np.float32)
    return R @ OPENXR_JOINT_TO_CANONICAL


@dataclass
class HandSample:
    """One hand at one poll."""

    joints: np.ndarray  # (26, 7) [x, y, z, qw, qx, qy, qz], sim world frame
    tracked: bool       # pose updated since the previous poll (False = lost / frozen / never seen)
    lost_polls: int     # consecutive polls without an update (0 when tracked)

    @property
    def wrist_pos(self) -> np.ndarray:
        return self.joints[WRIST, :3]

    @property
    def wrist_quat_wxyz(self) -> np.ndarray:
        return self.joints[WRIST, 3:7]


class _TrackingState:
    """Detects frozen hands: OpenXRDevice keeps the last valid pose when a hand is not tracked."""

    def __init__(self):
        self._last: Dict[str, Optional[np.ndarray]] = {s: None for s in SIDES}
        self._lost: Dict[str, int] = {s: 0 for s in SIDES}

    def reset(self):
        for s in SIDES:
            self._last[s] = None
            self._lost[s] = 0

    def update(self, side: str, joints: np.ndarray) -> HandSample:
        wrist = joints[WRIST]
        never_seen = not np.any(wrist[:3] != 0.0)
        frozen = self._last[side] is not None and np.array_equal(self._last[side], wrist)
        self._last[side] = wrist.copy()
        tracked = not (never_seen or frozen)
        self._lost[side] = 0 if tracked else self._lost[side] + 1
        return HandSample(joints=joints, tracked=tracked, lost_polls=self._lost[side])


class XrHandSource:
    """Isaac Lab OpenXRDevice wrapper (no retargeters; raw joints only).

    Must be created after the Kit app (with --xr) and the stage exist.
    """

    def __init__(
        self,
        anchor_pos: Sequence[float] = (0.0, -0.55, 0.0),
        anchor_rot: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        near_plane: float = 0.15,
        stream_log_every: int = 0,
    ):
        self._anchor_pos = tuple(float(v) for v in anchor_pos)
        self._anchor_rot = tuple(float(v) for v in anchor_rot)
        self._near_plane = float(near_plane)
        self._device = None
        self._tracking = _TrackingState()
        self._callbacks: Dict[str, Callable[[], None]] = {}
        self._log_every = int(stream_log_every)
        self._polls = 0
        self._dump = _JointDump.from_env()

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> bool:
        from isaaclab.devices.openxr import OpenXRDevice, OpenXRDeviceCfg, XrCfg
        from isaaclab.devices.retargeter_base import RetargeterBase

        xr_cfg = XrCfg(anchor_pos=self._anchor_pos, anchor_rot=self._anchor_rot, near_plane=self._near_plane)
        self._device = OpenXRDevice(OpenXRDeviceCfg(xr_cfg=xr_cfg))
        # Without retargeters OpenXRDevice requests no features; ask for hands + head explicitly.
        self._device._required_features.update(
            {RetargeterBase.Requirement.HAND_TRACKING, RetargeterBase.Requirement.HEAD_TRACKING}
        )
        for key, fn in self._callbacks.items():
            self._device.add_callback(key, fn)
        print(f"[XrHandSource] OpenXR device ready: anchor_pos={self._anchor_pos}, anchor_rot={self._anchor_rot}")
        return True

    def stop(self):
        if self._dump is not None:
            self._dump.flush()
        self._device = None

    def add_command_callback(self, key: str, fn: Callable[[], None]):
        """key: "START" | "STOP" | "RESET" (IsaacTeleop client buttons)."""
        self._callbacks[key] = fn
        if self._device is not None:
            self._device.add_callback(key, fn)

    def ensure_anchor(self):
        """Re-create the XR anchor prim if a scene rebuild removed it."""
        if self._device is None:
            return
        import omni.usd

        path = self._device._xr_anchor_headset_path
        stage = omni.usd.get_context().get_stage()
        if stage is not None and stage.GetPrimAtPath(path).IsValid():
            return
        from isaacsim.core.prims import SingleXFormPrim

        SingleXFormPrim(path, position=self._anchor_pos, orientation=self._anchor_rot)
        print(f"[XrHandSource] XR anchor re-created at {path}")

    # ── data ─────────────────────────────────────────────────────────────
    def poll(self) -> Dict[str, HandSample]:
        if self._device is None:
            return {}
        from isaaclab.devices.device_base import DeviceBase

        raw = self._device._get_raw_data()
        out: Dict[str, HandSample] = {}
        for side, target in (("right", DeviceBase.TrackingTarget.HAND_RIGHT),
                             ("left", DeviceBase.TrackingTarget.HAND_LEFT)):
            poses = raw.get(target)
            if poses is None:
                continue
            joints = np.stack([np.asarray(poses[n], dtype=np.float32) for n in HAND_JOINT_NAMES])
            out[side] = self._tracking.update(side, joints)
        self._after_poll(out)
        return out

    def reset_tracking(self):
        self._tracking.reset()

    def _after_poll(self, out: Dict[str, HandSample]):
        self._polls += 1
        if self._dump is not None:
            self._dump.append(out)
        if self._log_every > 0 and self._polls % self._log_every == 0:
            print(f"[xr] poll={self._polls:>6}  " + format_samples(out))


class NpzHandSource:
    """Replays a B2D_XR_DUMP recording (one recorded poll per poll(); loops at the end)."""

    def __init__(self, path: str, stream_log_every: int = 0):
        data = np.load(path)
        self._frames = {s: data[s] for s in SIDES if s in data.files}
        self._n = min(len(v) for v in self._frames.values())
        self._i = 0
        self._tracking = _TrackingState()
        self._log_every = int(stream_log_every)
        self._polls = 0
        print(f"[NpzHandSource] {path}: {self._n} polls, sides={list(self._frames)}")

    def start(self) -> bool:
        return True

    def stop(self):
        pass

    def add_command_callback(self, key: str, fn: Callable[[], None]):
        pass

    def ensure_anchor(self):
        pass

    def poll(self) -> Dict[str, HandSample]:
        out = {s: self._tracking.update(s, frames[self._i].astype(np.float32)) for s, frames in self._frames.items()}
        self._i = (self._i + 1) % self._n
        self._polls += 1
        if self._log_every > 0 and self._polls % self._log_every == 0:
            print(f"[xr-replay] poll={self._polls:>6}  " + format_samples(out))
        return out

    def reset_tracking(self):
        self._tracking.reset()


def format_samples(samples: Dict[str, HandSample]) -> str:
    parts = []
    for side in SIDES:
        s = samples.get(side)
        if s is None:
            parts.append(f"{side[0].upper()}: n/a")
            continue
        n_nonzero = int(np.count_nonzero(np.any(s.joints[:, :3] != 0.0, axis=1)))
        p = s.wrist_pos
        flag = "" if s.tracked else f" LOST({s.lost_polls})"
        parts.append(f"{side[0].upper()}: wrist=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}) nonzero={n_nonzero}/{NUM_XR_JOINTS}{flag}")
    return "  ".join(parts)


class _JointDump:
    """B2D_XR_DUMP=<file.npz>: record every poll's raw joints (right/left: (T, 26, 7)) for offline replay."""

    _FLUSH_EVERY = 60

    def __init__(self, path: str):
        self._path = path
        self._frames: Dict[str, list] = {s: [] for s in SIDES}
        print(f"[XrHandSource] recording raw hand joints to {path}")

    @classmethod
    def from_env(cls) -> Optional["_JointDump"]:
        path = os.environ.get("B2D_XR_DUMP")
        return cls(path) if path else None

    def append(self, samples: Dict[str, HandSample]):
        zeros = np.zeros((NUM_XR_JOINTS, 7), dtype=np.float32)
        for s in SIDES:
            self._frames[s].append(samples[s].joints if s in samples else zeros)
        if len(self._frames["right"]) % self._FLUSH_EVERY == 0:
            self.flush()

    def flush(self):
        if not self._frames["right"]:
            return
        np.savez(
            self._path,
            right=np.stack(self._frames["right"]),
            left=np.stack(self._frames["left"]),
            joint_names=np.array(HAND_JOINT_NAMES),
        )
