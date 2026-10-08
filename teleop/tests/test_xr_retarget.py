#!/usr/bin/env python3
"""
Offline checks for the Quest (OpenXR) finger path — no headset or Isaac Sim needed.

Builds synthetic OpenXR hands (26 joints, OpenXR joint-axis conventions: -Z toward the
fingertips, +Y out of the back of the hand) and checks, for every hand type:
  1. the robot fingertip vectors match the human ones for open and closed hands
     (catches a wrong wrist-frame convention: fingers would have to bend backwards),
  2. a fist pulls the robot fingertips toward the wrist,
  3. the fingertips do not depend on where/how the whole hand is held in the world.

Usage:
    source /workspace/bench2dex_env.sh
    python teleop/tests/test_xr_retarget.py [--hands wuji shadow ...]
"""

import argparse
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from teleop.retarget_bridge import RetargetBridge  # noqa: E402
from teleop.xr_hand_source import (  # noqa: E402
    HAND_JOINT_NAMES,
    canonical_wrist_frame,
    openxr_to_mediapipe,
)

UR5_HANDS = ("rh56dfx", "rh5dg2", "shadow", "schunk_svh", "wuji")

# Hand-local frame (right hand): +x fingers, +z back of hand, +y thumb side.
_MCP_Y = {"index": 0.025, "middle": 0.004, "ring": -0.016, "little": -0.034}
_MCP_X = {"index": 0.090, "middle": 0.093, "ring": 0.088, "little": 0.080}
_PHAL = {"index": (0.040, 0.024, 0.020), "middle": (0.045, 0.028, 0.022),
         "ring": (0.042, 0.026, 0.021), "little": (0.033, 0.019, 0.018)}


def _finger_chain(base, lengths, flex):
    """Joint positions from base along +x, each phalanx flexed by `flex` rad toward the palm (-z)."""
    pts, p, ang = [], np.asarray(base, float), 0.0
    for L in lengths:
        ang += flex
        p = p + L * np.array([np.cos(ang), 0.0, -np.sin(ang)])
        pts.append(p)
    return pts


def synthetic_hand_local(curl: float):
    """26 joint positions in the right-hand local frame; curl 0 = open, 1 = fist."""
    pos = {"wrist": np.zeros(3), "palm": np.array([0.045, -0.005, 0.0])}
    for f in _MCP_Y:
        mcp = np.array([_MCP_X[f], _MCP_Y[f], 0.0])
        pos[f"{f}_metacarpal"] = 0.35 * mcp
        pos[f"{f}_proximal"] = mcp
        names = (f"{f}_intermediate", f"{f}_distal", f"{f}_tip")
        for n, p in zip(names, _finger_chain(mcp, _PHAL[f], curl * 1.4)):
            pos[n] = p
    # Thumb: from CMC toward the index side, curling across the palm.
    cmc = np.array([0.025, 0.020, -0.012])
    pos["thumb_metacarpal"] = cmc
    d = np.array([0.6, 0.75, -0.25 - 0.5 * curl])
    d /= np.linalg.norm(d)
    side = np.array([0.0, -1.0, -0.3]) * curl
    pos["thumb_proximal"] = cmc + 0.045 * d
    pos["thumb_distal"] = pos["thumb_proximal"] + 0.032 * (d + 0.6 * side) / np.linalg.norm(d + 0.6 * side)
    pos["thumb_tip"] = pos["thumb_distal"] + 0.025 * (d + 1.2 * side) / np.linalg.norm(d + 1.2 * side)
    return pos


def to_openxr(curl: float, side: str, world_rot: Rotation, world_pos) -> np.ndarray:
    """(26, 7) OpenXR joints in world. Left hand = mirror of the right hand across local y."""
    local = synthetic_hand_local(curl)
    mirror = np.diag([1.0, -1.0, 1.0]) if side == "left" else np.eye(3)
    # Wrist joint axes in hand-local coords: Y = back of hand, Z = -fingers, X = Y x Z.
    Y, Z = np.array([0.0, 0.0, 1.0]), np.array([-1.0, 0.0, 0.0])
    R_joint_local = np.column_stack([np.cross(Y, Z), Y, Z])
    R_wrist_world = world_rot.as_matrix() @ R_joint_local
    qx, qy, qz, qw = Rotation.from_matrix(R_wrist_world).as_quat()
    out = np.zeros((len(HAND_JOINT_NAMES), 7), dtype=np.float32)
    for i, name in enumerate(HAND_JOINT_NAMES):
        out[i, :3] = world_rot.apply(mirror @ local[name]) + world_pos
        out[i, 3:] = (qw, qx, qy, qz)  # only the wrist orientation is used
    return out


def retarget(bridge, joints, side):
    """Settled finger qpos + robot wrist->tip vectors and their mean error to the human targets."""
    wrist = HAND_JOINT_NAMES.index("wrist")
    bridge.reset()
    q = None
    for _ in range(30):  # let the warm-started optimizer + low-pass filter settle
        q = bridge.retarget_keypoints(openxr_to_mediapipe(joints), canonical_wrist_frame(joints[wrist, 3:]))
    robot = bridge.last_robot_wrist_to_tips.copy()
    err = float(np.linalg.norm(robot - bridge.last_human_wrist_to_tips, axis=1).mean())
    return q, robot, err


def check_hand(hand_type: str, rng) -> list:
    failures = []
    for side in ("right", "left"):
        bridge = RetargetBridge(hand_type, side, input_device="xr")
        ident = Rotation.identity()
        q_open, tips_open, err_open = retarget(bridge, to_openxr(0.0, side, ident, np.zeros(3)), side)
        q_fist, tips_fist, err_fist = retarget(bridge, to_openxr(1.0, side, ident, np.zeros(3)), side)
        # 1) the robot fingertips follow the human ones (a wrong wrist frame gives 6-30 cm errors;
        #    Schunk SVH cannot close a full fist and stays ~4 cm off)
        if max(err_open, err_fist) > 0.05:
            failures.append(f"{hand_type}/{side}: fingertip vector error open={err_open*100:.1f}cm fist={err_fist*100:.1f}cm")
        # 2) a fist pulls the four fingertips toward the wrist (sign-agnostic flexion check)
        shrink = np.linalg.norm(tips_fist[1:], axis=1).mean() / np.linalg.norm(tips_open[1:], axis=1).mean()
        if shrink > 0.85:
            failures.append(f"{hand_type}/{side}: fist does not close the fingers (tip length ratio {shrink:.2f})")
        # 3) invariance to the world pose of the whole hand (fingertips, not qpos: redundant hands
        #    such as Shadow reach the same fingertips with slightly different joint splits)
        worst = 0.0
        for _ in range(5):
            R = Rotation.random(random_state=rng)
            p = rng.uniform(-1.0, 1.0, size=3)
            _, tips_rot, _ = retarget(bridge, to_openxr(1.0, side, R, p), side)
            worst = max(worst, float(np.max(np.linalg.norm(tips_rot - tips_fist, axis=1))))
        if worst > 0.005:
            failures.append(f"{hand_type}/{side}: fingertips change with world pose (max {worst*1000:.1f} mm)")
        print(f"  {hand_type:10s} {side:5s} tip err open={err_open*100:.1f}cm fist={err_fist*100:.1f}cm "
              f"fist/open tip length={shrink:.2f} world-pose max tip shift={worst*1000:.1f}mm")
    return failures


def test_xr_retarget(hands=UR5_HANDS):
    rng = np.random.default_rng(0)
    failures = []
    for h in hands:
        failures += check_hand(h, rng)
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", nargs="*", default=list(UR5_HANDS))
    args = ap.parse_args()
    try:
        test_xr_retarget(args.hands)
    except AssertionError as e:
        print("FAIL\n" + str(e))
        sys.exit(1)
    print("PASS")
