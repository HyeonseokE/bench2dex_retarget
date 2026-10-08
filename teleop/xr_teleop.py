"""
xr_teleop.py — Meta Quest 3 (OpenXR hand tracking) teleoperation for Bench2Dex.

Replaces the Manus SHM + iPhone ARKit inputs with one OpenXR source and reuses the rest:

  XrHandSource (26 joints/hand, sim world frame)
    ├ fingers: 21 MediaPipe keypoints + canonical wrist frame → RetargetBridge.retarget_keypoints (DexPilot)
    └ wrists : relative mapping → ArmIKController (one per arm)
  XrIsaacLabBridge.update(hold_targets)  — same API as TeleopIsaacLabBridge, so main.py's loop,
  homing and the collector are unchanged.

Wrist mapping (DexVerse SimpleRelativeRetargeter "quat_absolute" mode, i.e. arm robots):
  at anchor time  : human wrist (p_h0, R_h0), robot palm p_r0 = ee_pos + R_ee0 @ ee_offset, R_ee0
  afterwards      : palm target  = p_r0 + pos_scale * (p_h - p_h0)
                    EE rotation  = (R_h @ R_h0^T) @ R_ee0          (world-frame delta)
  The rotation goes to IK as a quaternion — no Euler angles, so no gimbal lock / ±180° wrap.

Anchoring (= clutch) happens
  * after every homing (main.py calls reset_input_state()), i.e. on headset START / STOP / RESET,
  * when a hand reappears after being lost for more than relock_after_lost_polls polls
    (drop the hand out of view, move it, bring it back: the robot does not jump),
  * after too many consecutive rejected jumps.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from .arm_ik_controller import ArmIKController, detect_arm_type
from .isaaclab_bridge import TeleopIsaacLabBridge
from .math_utils import rotmat_to_quat_wxyz, quat_wxyz_to_rotmat
from .retarget_bridge import RetargetBridge, _HAND_TYPE_TO_CONFIG
from .xr_hand_source import SIDES, HandSample, canonical_wrist_frame, openxr_to_mediapipe


class XrTeleopController:
    """OpenXR hands → finger joint targets (+ filtered wrist poses for the arm bridge).

    Exposes the attributes TeleopIsaacLabBridge reads from TeleopController
    (_hand_type, _enable_right/_enable_left, _bridge_right/_bridge_left, step(), reset_filters()).
    """

    def __init__(
        self,
        source,
        hand_type: str,
        enable_right: bool = True,
        enable_left: bool = True,
        wrist_smoothing: float = 0.5,
    ):
        """
        Args:
            source: XrHandSource or NpzHandSource.
            hand_type: key of _HAND_TYPE_TO_CONFIG.
            wrist_smoothing: 0 = no filter; closer to 1 = smoother wrist pose (EMA / slerp per poll).
        """
        if hand_type not in _HAND_TYPE_TO_CONFIG:
            raise ValueError(f"Unknown hand type '{hand_type}'. Available: {list(_HAND_TYPE_TO_CONFIG)}")
        if not 0.0 <= wrist_smoothing < 1.0:
            raise ValueError(f"wrist_smoothing must be in [0, 1), got {wrist_smoothing}")
        self._source = source
        self._hand_type = hand_type
        self._enable_right = enable_right
        self._enable_left = enable_left
        self._smoothing = float(wrist_smoothing)
        self._bridge_right: Optional[RetargetBridge] = None
        self._bridge_left: Optional[RetargetBridge] = None
        self._started = False
        self._samples: Dict[str, HandSample] = {}
        self._wrist_pos: Dict[str, Optional[np.ndarray]] = {s: None for s in SIDES}
        self._wrist_rot: Dict[str, Optional[Rotation]] = {s: None for s in SIDES}

    @property
    def hand_type(self) -> str:
        return self._hand_type

    @property
    def source(self):
        return self._source

    def start(self, wait_timeout: float = 0.0) -> bool:
        if self._started:
            return True
        if not self._source.start():
            return False
        if self._enable_right:
            self._bridge_right = RetargetBridge(self._hand_type, "right", input_device="xr")
        if self._enable_left:
            self._bridge_left = RetargetBridge(self._hand_type, "left", input_device="xr")
        self._started = True
        print(f"[XrTeleopController] Started: hand_type={self._hand_type}, "
              f"right={'ON' if self._enable_right else 'OFF'}, left={'ON' if self._enable_left else 'OFF'}")
        return True

    def stop(self):
        if self._started:
            self._source.stop()
        self._started = False
        self._bridge_right = None
        self._bridge_left = None

    def _bridge(self, side: str) -> Optional[RetargetBridge]:
        return self._bridge_right if side == "right" else self._bridge_left

    def step(self) -> Optional[Dict[str, np.ndarray]]:
        """Poll the headset; returns {"right"/"left": finger qpos} for tracked hands, or None."""
        if not self._started:
            return None
        self._samples = self._source.poll()
        result: Dict[str, np.ndarray] = {}
        for side in SIDES:
            sample = self._samples.get(side)
            if sample is None or not sample.tracked:
                continue
            self._filter_wrist(side, sample)
            bridge = self._bridge(side)
            if bridge is None:
                continue
            frame = canonical_wrist_frame(sample.wrist_quat_wxyz)
            q = bridge.retarget_keypoints(openxr_to_mediapipe(sample.joints), frame)
            if q is not None:
                result[side] = q
        return result if result else None

    def _filter_wrist(self, side: str, sample: HandSample):
        pos = sample.wrist_pos.astype(np.float64)
        q = sample.wrist_quat_wxyz.astype(np.float64)
        rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        prev_pos, prev_rot = self._wrist_pos[side], self._wrist_rot[side]
        if prev_pos is None or sample.lost_polls > 0 or self._smoothing == 0.0:
            self._wrist_pos[side], self._wrist_rot[side] = pos, rot
            return
        a = 1.0 - self._smoothing  # weight of the new sample
        self._wrist_pos[side] = prev_pos + a * (pos - prev_pos)
        self._wrist_rot[side] = Slerp([0.0, 1.0], Rotation.concatenate([prev_rot, rot]))(a)

    def wrist_pose(self, side: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Filtered wrist (position (3,), rotation matrix (3, 3)) in the sim world frame, or None."""
        if self._wrist_pos[side] is None:
            return None
        return self._wrist_pos[side].copy(), self._wrist_rot[side].as_matrix()

    def lost_polls(self, side: str) -> Optional[int]:
        """0 = tracked this poll; None = no data for this hand yet."""
        sample = self._samples.get(side)
        return None if sample is None else sample.lost_polls

    def reset_filters(self):
        for bridge in (self._bridge_right, self._bridge_left):
            if bridge is not None:
                bridge.reset()
        for side in SIDES:
            self._wrist_pos[side] = None
            self._wrist_rot[side] = None


class _WristAnchor:
    def __init__(self, p_h0, R_h0, p_r0, R_ee0):
        self.p_h0, self.R_h0, self.p_r0, self.R_ee0 = p_h0, R_h0, p_r0, R_ee0
        self.last_target: Optional[np.ndarray] = None
        self.last_R_target: Optional[np.ndarray] = None
        self.rejected = 0


class XrIsaacLabBridge(TeleopIsaacLabBridge):
    """TeleopIsaacLabBridge with OpenXR wrists (relative mapping) instead of ARKit."""

    def __init__(
        self,
        teleop: XrTeleopController,
        articulation,
        pos_scale: float = 1.0,
        relock_after_lost_polls: int = 6,
        max_target_step_m: float = 0.08,
        max_rejected_steps: int = 5,
        log_every: int = 0,
    ):
        """
        Args:
            pos_scale: human wrist displacement → robot palm displacement.
            relock_after_lost_polls: re-anchor a hand that was lost this many polls (20 Hz → 0.3 s).
            max_target_step_m: palm target jump per poll treated as a tracking glitch (frame skipped).
            max_rejected_steps: consecutive glitches after which the hand is re-anchored.
            log_every: print palm tracking error (target vs. actual position / rotation) every N updates.
        """
        # The base class would create ARKit readers when arm teleop is on; we set up the arms ourselves.
        super().__init__(teleop, articulation, enable_arm_teleop=False)
        self._enable_arm_teleop = True
        self._arm_type_right = detect_arm_type(self._joint_names, side="right")
        self._arm_type_left = detect_arm_type(self._joint_names, side="left")
        self._pos_scale = float(pos_scale)
        self._relock_after = int(relock_after_lost_polls)
        self._max_step = float(max_target_step_m)
        self._max_rejected = int(max_rejected_steps)
        self._anchors: Dict[str, Optional[_WristAnchor]] = {s: None for s in SIDES}
        self._log_every = int(log_every)
        self._err: Dict[str, List[Tuple[float, float]]] = {s: [] for s in SIDES}
        self._n_updates = 0
        print(f"[XrIsaacLabBridge] arms: right={self._arm_type_right}, left={self._arm_type_left}, "
              f"pos_scale={self._pos_scale}")

    # ── arm IK setup (base version requires an ARKit reader) ─────────────
    def _init_arm_ik_side(self, side: str):
        arm_type = self._arm_type_right if side == "right" else self._arm_type_left
        if arm_type is None:
            return
        try:
            ik = ArmIKController(
                arm_type, num_envs=self._articulation.num_instances, device=str(self._articulation.device),
                side=side, hand_type=self._teleop._hand_type,
            )
            name_to_idx = {n: i for i, n in enumerate(self._joint_names)}
            joint_indices = [name_to_idx[j] for j in ik.joint_names if j in name_to_idx]
            body_names = list(self._articulation.data.body_names)
            if ik.ee_body not in body_names:
                raise ValueError(f"EE body '{ik.ee_body}' not in body_names {body_names}")
            ee_body_idx = body_names.index(ik.ee_body)
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"[XrIsaacLabBridge] {side.title()} arm IK init failed: {e}")
            return
        if side == "right":
            self._arm_ik_right, self._arm_right_joint_indices, self._arm_right_ee_body_idx = ik, joint_indices, ee_body_idx
        else:
            self._arm_ik_left, self._arm_left_joint_indices, self._arm_left_ee_body_idx = ik, joint_indices, ee_body_idx
        print(f"[XrIsaacLabBridge] {side.title()} arm IK ready, joints={ik.joint_names}, ee={ik.ee_body}")

    def _arm(self, side: str):
        if side == "right":
            return self._arm_ik_right, self._arm_right_joint_indices, self._arm_right_ee_body_idx
        return self._arm_ik_left, self._arm_left_joint_indices, self._arm_left_ee_body_idx

    # ── per-frame arm update ─────────────────────────────────────────────
    def _update_arm_ik(self, target) -> bool:
        applied = False
        for side in SIDES:
            applied = self._update_arm_side(side, target) or applied
        self._n_updates += 1
        if self._log_every > 0 and self._n_updates % self._log_every == 0:
            self._log_tracking_error()
        return applied

    def _log_tracking_error(self):
        parts = []
        for side in SIDES:
            errs = self._err[side]
            if not errs:
                parts.append(f"{side[0].upper()}: -")
                continue
            e = np.asarray(errs)
            parts.append(f"{side[0].upper()}: pos {e[:, 0].mean() * 100:.1f}/{e[:, 0].max() * 100:.1f}cm "
                         f"rot {e[:, 1].mean():.1f}/{e[:, 1].max():.1f}deg")
            errs.clear()
        print("[xr-track] palm target vs actual (mean/max): " + "  ".join(parts))

    def _update_arm_side(self, side: str, target) -> bool:
        ik, arm_cols, ee_idx = self._arm(side)
        if ik is None:
            return False
        lost = self._teleop.lost_polls(side)
        if lost is None:
            return False
        if lost > 0:
            # Hold the last arm target; a long loss re-anchors on reappearance.
            if lost >= self._relock_after:
                self._anchors[side] = None
            return False
        pose = self._teleop.wrist_pose(side)
        if pose is None:
            return False
        p_h, R_h = pose

        art = self._articulation
        ee_pos_w = art.data.body_pos_w[:, ee_idx]
        ee_quat_w = art.data.body_quat_w[:, ee_idx]
        anchor = self._anchors[side]
        if anchor is None:
            R_ee0 = quat_wxyz_to_rotmat(ee_quat_w[0].cpu().numpy())
            p_r0 = ee_pos_w[0].cpu().numpy().astype(np.float64)
            ee_offset = getattr(ik, "_ee_offset", None)
            if ee_offset is not None:
                p_r0 = p_r0 + R_ee0 @ ee_offset
            self._anchors[side] = _WristAnchor(p_h, R_h, p_r0, R_ee0)
            print(f"[XrIsaacLabBridge] {side} wrist anchored: human={np.round(p_h, 3)}, robot palm={np.round(p_r0, 3)}")
            return False

        palm_target = anchor.p_r0 + self._pos_scale * (p_h - anchor.p_h0)
        if anchor.last_target is not None and np.linalg.norm(palm_target - anchor.last_target) > self._max_step:
            anchor.rejected += 1
            if anchor.rejected >= self._max_rejected:
                print(f"[XrIsaacLabBridge] {side}: {anchor.rejected} consecutive jumps, re-anchoring")
                self._anchors[side] = None
            return False
        anchor.rejected = 0
        R_target = (R_h @ anchor.R_h0.T) @ anchor.R_ee0
        if anchor.last_target is not None:
            self._record_tracking_error(side, ik, ee_pos_w, ee_quat_w, anchor)
        anchor.last_target = palm_target
        anchor.last_R_target = R_target

        joint_pos_des = ik.compute(
            target_pos=palm_target, target_quat=rotmat_to_quat_wxyz(R_target),
            current_joint_pos=art.data.joint_pos[:, arm_cols],
            ee_pos_w=ee_pos_w, ee_quat_w=ee_quat_w,
            root_pos_w=art.data.root_pos_w, root_quat_w=art.data.root_quat_w,
        )
        target[0, arm_cols] = joint_pos_des[0, :len(arm_cols)]
        self._visualize_arm(side, palm_target, ee_pos_w)
        return True

    def _record_tracking_error(self, side: str, ik, ee_pos_w, ee_quat_w, anchor: "_WristAnchor"):
        """Error of the current palm pose w.r.t. the previous update's target (one update of lag)."""
        R_ee = quat_wxyz_to_rotmat(ee_quat_w[0].cpu().numpy())
        palm = ee_pos_w[0].cpu().numpy().astype(np.float64)
        ee_offset = getattr(ik, "_ee_offset", None)
        if ee_offset is not None:
            palm = palm + R_ee @ ee_offset
        pos_err = float(np.linalg.norm(palm - anchor.last_target))
        rot_err = float(np.degrees(Rotation.from_matrix(R_ee @ anchor.last_R_target.T).magnitude()))
        self._err[side].append((pos_err, rot_err))

    def _visualize_arm(self, side: str, palm_target: np.ndarray, ee_pos_w):
        tgt_marker = self._target_marker if side == "right" else self._left_target_marker
        ee_marker = self._ee_marker if side == "right" else self._left_ee_marker
        device = self._articulation.device
        if tgt_marker is not None:
            tgt_marker.visualize(translations=torch.tensor(palm_target, dtype=torch.float32, device=device).unsqueeze(0))
        if ee_marker is not None:
            ee_marker.visualize(translations=ee_pos_w)

    # ── runtime API used by main.py ──────────────────────────────────────
    def reset_arm_anchor(self):
        """Next tracked frame of each hand becomes the new reference (called after homing)."""
        for side in SIDES:
            self._anchors[side] = None

    def reset_articulation(self, new_articulation) -> None:
        super().reset_articulation(new_articulation)
        self.reset_arm_anchor()
        self._teleop.source.ensure_anchor()

    def stop(self):
        self._teleop.stop()
        self._started = False

    @property
    def arm_sides(self) -> List[str]:
        return [s for s in SIDES if self._arm(s)[0] is not None]
