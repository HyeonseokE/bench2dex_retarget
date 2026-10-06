"""The five UR5 dual-arm hands, as Bench2Dex defines them.

Nothing about a robot's physics is re-typed here. ``spawn_robot`` runs Bench2Dex's own spawner, so
actuator gains, home pose, drive-type fixes, hand friction and gravity compensation are exactly the
benchmark's. What this module adds is the kinematic vocabulary the retargeting needs and Bench2Dex
does not spell out in one place:

  tips      five fingertip points per hand, ALWAYS ordered thumb, index, middle, ring, little, so
            finger k of the source maps onto finger k of the target. A point is (body, offset in
            the body frame). Bodies come from Bench2Dex/teleop/hand_cfgs/*.yml where the USD has
            them; RH5DG2 and Schunk distal links carry no tip body on both sides (Schunk's left
            tips were lost to a duplicate link name in its URDF), so for those the offset is the
            distal mesh's farthest vertex, read from the USDs (2026-10-05).
  wrist     the hand base body (teleop wrist_link_name); the SPIDER wrist residual acts on it.
  flange    tool0, softly kept at the source flange pose during kinematic retargeting.
  actuated  hand joints teleop drives. The rest either follow a mimic constraint in the USD or
            are locked (Shadow THJ3/LFJ5); retarget.coupling models them for IK.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass, field

import yaml

from . import paths

ARM = {
    "right": ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
    "left": ["L_arm_shoulder_pan_joint", "L_arm_shoulder_lift_joint", "L_arm_elbow_joint",
             "L_arm_wrist_1_joint", "L_arm_wrist_2_joint", "L_arm_wrist_3_joint"],
}
FLANGE = {"right": "tool0", "left": "L_arm_tool0"}
SIDES = ("right", "left")
FINGERS = ("thumb", "index", "middle", "ring", "little")


@dataclass(frozen=True)
class RobotSpec:
    name: str                         # short name used on the command line
    key: str                          # Bench2Dex robot_key
    teleop_cfg: str                   # Bench2Dex/teleop/hand_cfgs/<teleop_cfg>.yml
    wrist: dict                       # side -> body
    tips: dict                        # side -> [(body, (x, y, z)), ...] thumb..little
    hand_side: str                    # regex group: a hand joint belongs to "left" if it matches
    locked: tuple = ()                # hand joints held at 0
    actuated_from_teleop: bool = True  # else: every non-locked hand joint is actuated
    extra: dict = field(default_factory=dict)

    def side_of(self, joint: str) -> str:
        return "left" if re.match(self.hand_side, joint) else "right"

    def actuated(self, side: str, hand_joints: list[str]) -> list[str]:
        if not self.actuated_from_teleop:
            return [j for j in hand_joints if j not in self.locked]
        cfg = yaml.safe_load(open(paths.BENCH2DEX / "teleop" / "hand_cfgs" / f"{self.teleop_cfg}.yml"))
        names = cfg["retargeting"][side]["target_joint_names"]
        missing = [n for n in names if n not in hand_joints]
        assert not missing, f"{self.name}: teleop joints {missing} not in the articulation"
        return list(names)


def _same(side_names):
    return {s: [(n, (0.0, 0.0, 0.0)) for n in side_names[s]] for s in SIDES}


ROBOTS: dict[str, RobotSpec] = {
    "rh56dfx": RobotSpec(
        name="rh56dfx", key="multi_ur5_rh56dfx_with_flange", teleop_cfg="rh56dfx",
        wrist={"right": "r_base_link", "left": "l_base_link"},
        tips=_same({s: [f"{s}_{f}_tip" for f in FINGERS] for s in SIDES}),
        hand_side=r"left_"),
    "rh5dg2": RobotSpec(
        name="rh5dg2", key="multi_ur5_rh5dg2_with_flange", teleop_cfg="rh5dg2",
        wrist={"right": "right_hand_base", "left": "left_hand_base"},
        # teleop uses the force-sensor pads as fingertips; they sit on the distal pads
        tips=_same({s: [f"{s}_{f}_force_sensor" for f in ("thumb", "index", "middle", "ring", "pinky")]
                    for s in SIDES}),
        hand_side=r"left_"),
    "shadow": RobotSpec(
        name="shadow", key="multi_ur5_shadow_hand_with_flange", teleop_cfg="shadow",
        wrist={"right": "palm", "left": "l_palm"},
        tips=_same({"right": ["thtip", "fftip", "mftip", "rftip", "lftip"],
                    "left": ["l_thtip", "l_fftip", "l_mftip", "l_rftip", "l_lftip"]}),
        hand_side=r"l_", locked=("THJ3", "LFJ5", "l_THJ3", "l_LFJ5"), actuated_from_teleop=False),
    "schunk": RobotSpec(
        name="schunk", key="multi_ur5_schunk_hand_with_flange", teleop_cfg="schunk_svh",
        wrist={"right": "right_hand_base_link", "left": "left_hand_base_link"},
        # distal links: c thumb, t index, s middle, r ring, q little (joint chains of the USD)
        tips={s: [(f"{s}_hand_c", (0.0278, 0.0, 0.0)), (f"{s}_hand_t", (0.0225, 0.0, 0.0)),
                  (f"{s}_hand_s", (0.0225, 0.0, 0.0)), (f"{s}_hand_r", (0.0225, 0.0, 0.0)),
                  (f"{s}_hand_q", (0.0225, 0.0, 0.0))] for s in SIDES},
        hand_side=r"left_"),
    "wuji": RobotSpec(
        name="wuji", key="multi_ur5_wuji_with_flange", teleop_cfg="wuji",
        wrist={"right": "right_palm_link", "left": "left_palm_link"},
        tips=_same({s: [f"{s}_finger{i}_tip_link" for i in range(1, 6)] for s in SIDES}),
        hand_side=r"left_", actuated_from_teleop=False),
}
BY_KEY = {r.key: r for r in ROBOTS.values()}


def get(name_or_key: str) -> RobotSpec:
    return ROBOTS.get(name_or_key) or BY_KEY[name_or_key]


ENV0_PRIM = "/World/envs/env_0/Robot"
REGEX_PRIM = "/World/envs/env_.*/Robot"


def spawn_robot(spec: RobotSpec, table_size, robot_mount_height: float) -> dict:
    """Run Bench2Dex's spawner inside a DirectRLEnv ``_setup_scene`` (only env_0 exists yet).

    The spawner builds its ArticulationCfg against a module constant ``ROBOT_PRIM_PATH`` and then
    fixes drive types / binds friction on that prim. Pointing the constant at env_0 and wrapping
    ``Articulation`` so the asset itself is created on the env regex gives one articulation over
    all clones, with every post-spawn fix applied to the env_0 template before it is cloned.
    Returns the spawner's runtime dict; ``pre_step_hooks`` (gravity compensation) then act on the
    whole batch.
    """
    paths.use_bench2dex()
    from isaaclab.assets import Articulation

    from robots import ROBOT_SPAWNERS  # Bench2Dex

    mod_name, fn_name = ROBOT_SPAWNERS[spec.key]
    mod = importlib.import_module(f"robots.{mod_name}")
    saved = mod.ROBOT_PRIM_PATH, mod.Articulation

    def _regex_articulation(cfg):
        return Articulation(cfg.replace(prim_path=REGEX_PRIM))

    mod.ROBOT_PRIM_PATH, mod.Articulation = ENV0_PRIM, _regex_articulation
    try:
        out = getattr(mod, fn_name)(table_size=tuple(table_size), table_height=robot_mount_height,
                                    task_dir=str(paths.BENCH2DEX / "scenes"))
    finally:
        mod.ROBOT_PRIM_PATH, mod.Articulation = saved
    return out
