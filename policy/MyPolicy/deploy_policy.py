"""Template adapter for evaluating your own policy on Bench2Dex.

The policy server (script/policy_model_server.py) imports this module as
``policy.MyPolicy.deploy_policy`` and uses the default session in
script/policy_sessions.py, which calls:

    model = get_model(usr_args)          # once, at server start
    reset_model(model)                   # at the start of every episode
    obs = encode_obs(raw_observation)    # every policy step
    actions = model.get_action(obs)      # -> (T, action_dim) chunk, executed open-loop

``usr_args`` is deploy_policy.yml merged with the CLI overrides passed by
eval_double_env.sh (ckpt_dir, ckpt_name, task_name, seed, use_active_dof, ...).

Raw observation sent by run_policy.py (see ``_build_remote_policy_obs``):
    raw["observation"][cam]["rgb"]  uint8 HxWx3, cam in
        cam_wrist_left, cam_wrist_right, cam_stereo_left, cam_stereo_right,
        cam_overhead, cam_chest (aliases: head_camera/left_camera/right_camera/front_camera)
    raw["joint_action"]["qpos"]     float32 (full_dof,) robot joint positions
    raw["language"]                 task instruction string

Action convention (same as the released ACT/DP/pi0.5/GR00T checkpoints):
absolute joint-position targets. With ``use_active_dof: true`` the policy sees
and outputs only the active DoFs (e.g. 24 for multi_ur5_rh56dfx_with_flange,
see robots/active_dof_maps.yml); the client expands them back to full DoF.

Training data convention: truncate each episode at ``meta/homing_start_sim_step``
(see README "Policy Data Preparation").
"""

import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

CAMERA_NAMES = ["cam_wrist_right", "cam_wrist_left", "cam_stereo_left", "cam_stereo_right"]

_ACTIVE_DOF_INFO = None


def _resolve_active_dof(usr_args: dict):
    """Find the robot from usr_args['robot_key'] or the ckpt_dir path
    (.../policy_ckpt/<task>/<robot_key>/<policy>/)."""
    global _ACTIVE_DOF_INFO
    from robots import ROBOT_SPAWNERS
    from robots.active_dof_utils import get_active_dof_info

    robot_key = usr_args.get("robot_key") or None
    if robot_key is None and usr_args.get("ckpt_dir"):
        for part in os.path.normpath(str(usr_args["ckpt_dir"])).split(os.sep)[::-1]:
            if part in ROBOT_SPAWNERS:
                robot_key = part
                break
    if robot_key is None:
        raise ValueError("Cannot determine robot_key; pass it in deploy_policy.yml")
    info = get_active_dof_info(robot_key)
    _ACTIVE_DOF_INFO = info if usr_args.get("use_active_dof", True) else None
    return robot_key, info


def encode_obs(observation: dict) -> dict:
    qpos = np.asarray(observation["joint_action"]["qpos"], dtype=np.float32)
    if _ACTIVE_DOF_INFO is not None and qpos.shape[-1] == _ACTIVE_DOF_INFO.full_dof:
        from robots.active_dof_utils import select_active
        qpos = select_active(qpos, _ACTIVE_DOF_INFO)
    return {
        "images": {cam: observation["observation"][cam]["rgb"] for cam in CAMERA_NAMES},
        "qpos": qpos,
        "instruction": observation.get("language", ""),
    }


class MyPolicy:
    def __init__(self, usr_args: dict):
        self.robot_key, info = _resolve_active_dof(usr_args)
        self.action_dim = info.active_dof if usr_args.get("use_active_dof", True) else info.full_dof
        self.device = usr_args.get("device", "cuda:0")
        ckpt_path = os.path.join(usr_args["ckpt_dir"], usr_args["ckpt_name"])
        # TODO: build your network and load weights from ckpt_path.
        self.net = None
        print(f"[MyPolicy] robot={self.robot_key} action_dim={self.action_dim} ckpt={ckpt_path}", flush=True)

    def reset(self):
        # TODO: clear history buffers / temporal ensembling state.
        pass

    def get_action(self, obs: dict) -> np.ndarray:
        # TODO: replace with real inference; must return (T, action_dim) absolute joint targets.
        # Placeholder: hold the current pose for one step.
        return obs["qpos"][None, :].astype(np.float32)


def get_model(usr_args: dict):
    return MyPolicy(usr_args)


def reset_model(model):
    model.reset()
