"""One Bench2Dex episode as an Isaac Lab DirectRLEnv with N identical copies.

All copies hold the same episode: SPIDER broadcasts one simulator state to N envs and rolls out N
action samples, the kinematic stage solves one frame per env, and the reference stage uses one.
Because the episode is fixed, the scene is built for it directly -- table at the episode's
(generalized) height, objects at their recorded first-frame poses -- with Bench2Dex's own helpers:

  table / support table   build.table_geometry, as build.scene_builder._spawn_table spawns them
  robot                   b2dr.robots.spawn_robot (the benchmark spawner, unmodified)
  objects                 build.spawners._build_usd_cfg + build.object_initial_state for
                          articulations (actuators, joint limits, fix_root_link)

Physics matches main.py: dt 1/60, CCD, stabilization; 20 Hz control = recording rate (decimation 3).

Pose conventions: env-local frame == Bench2Dex world frame (env origins are subtracted on read and
added on write), quaternions wxyz.
"""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from . import paths, robots
from .task import load_episode, load_scene

PHYSICS_DT = 1.0 / 60.0


@configclass
class B2DEnvCfg(DirectRLEnvCfg):
    task: str = ""                  # task number or scene name
    episode: int = 0
    robot: str = ""                 # robots.ROBOTS name
    with_objects: bool = True       # False: robot only (kinematic stage)
    gravity: bool = True

    decimation: int = 3
    episode_length_s: float = 1.0e6  # never times out; the caller decides when an episode ends
    sim: SimulationCfg = SimulationCfg(
        dt=PHYSICS_DT, render_interval=3,
        physx=PhysxCfg(enable_ccd=True, enable_stabilization=True, bounce_threshold_velocity=0.01,
                       gpu_max_rigid_contact_count=2**23, gpu_max_rigid_patch_count=2**22,
                       gpu_found_lost_pairs_capacity=2**22, gpu_found_lost_aggregate_pairs_capacity=2**25,
                       gpu_total_aggregate_pairs_capacity=2**22),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)
    action_space: int = 1
    observation_space: int = 1
    state_space: int = 0


class B2DEnv(DirectRLEnv):
    """Scene + state access. Control (actions) lives in subclasses; the base env only holds targets."""

    cfg: B2DEnvCfg

    def __init__(self, cfg: B2DEnvCfg, render_mode=None, **kwargs):
        if not cfg.gravity:
            cfg.sim.gravity = (0.0, 0.0, 0.0)
        self.spec = robots.get(cfg.robot)
        self.scene_def = load_scene(_scene_name(cfg.task))
        self.episode = load_episode(cfg.task, cfg.episode)
        self.table_size, self.table_heights, self.table_geom = self.episode.table_heights(self.scene_def)
        self.objects: dict = {}
        self.pre_step_hooks: list = []
        super().__init__(cfg, render_mode, **kwargs)
        self._init_robot_indices()
        self._init_object_bodies()
        self.q_target = self.robot.data.joint_pos.clone()
        self.frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        if self.cfg.with_objects:          # robot-only scenes (kinematics) have nothing to collide with
            g = self.table_geom
            table = sim_utils.MeshCuboidCfg(
                size=g.spawn_size, collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.30, 0.20)))
            table.func("/World/envs/env_0/Table", table, translation=g.center)
            from build.table_geometry import resolve_robot_support_table_geometry  # Bench2Dex
            sup = resolve_robot_support_table_geometry()
            if sup is not None:
                sc = sim_utils.MeshCuboidCfg(size=sup.size, collision_props=sim_utils.CollisionPropertiesCfg())
                sc.func("/World/envs/env_0/RobotSupportTable", sc, translation=sup.center)

        runtime = robots.spawn_robot(self.spec, self.table_size, self.table_heights.robot_mount_height)
        self.robot = runtime["interactive_objects"]["global_robot"]
        self.pre_step_hooks = list(runtime.get("pre_step_hooks", []))
        self.robot_pose = runtime.get("robot_pose", {})

        if self.cfg.with_objects:
            for o in self.scene_def.objects:
                self.objects[o.obj_id] = self._spawn_object(o)

        sim_utils.spawn_ground_plane("/World/ground", sim_utils.GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        for k, a in self.objects.items():
            (self.scene.articulations if isinstance(a, Articulation) else self.scene.rigid_objects)[k] = a
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    def _spawn_object(self, o):
        from build.articulation_limits import apply_articulation_joint_limits
        from build.object_initial_state import (build_articulation_actuator_cfgs_from_specs,
                                                resolve_articulation_spawn_config)
        from build.spawners import _build_usd_cfg

        ep = self.episode
        p0 = ep.obj_pose[o.obj_id][0]
        pos, rot = tuple(float(v) for v in p0[:3]), tuple(float(v) for v in p0[3:7])
        prim = f"/World/envs/env_.*/{o.obj_id}"
        if o.body_type == "articulation":
            spawn = resolve_articulation_spawn_config(o.asset_spec, o.obj_spec)
            usd = _build_usd_cfg(path=o.usd_path, scale=o.scale, body="articulation", spawn=spawn)
            names = ep.obj_joint_names[o.obj_id]
            jp = {n: float(v) for n, v in zip(names, ep.obj_qpos[o.obj_id][0])}
            art = Articulation(ArticulationCfg(
                prim_path=prim, spawn=usd,
                init_state=ArticulationCfg.InitialStateCfg(pos=pos, rot=rot, joint_pos=jp),
                actuators=build_articulation_actuator_cfgs_from_specs(spawn["actuators"])))
            apply_articulation_joint_limits(f"/World/envs/env_0/{o.obj_id}", spawn.get("joint_limits"),
                                            log_prefix=f"b2dr:{o.obj_id}")
            return art
        spawn = {}
        for src in (o.asset_spec.get("spawn"), o.obj_spec.get("spawn")):
            if isinstance(src, dict):
                for k, v in src.items():
                    spawn[k] = {**spawn.get(k, {}), **v} if isinstance(v, dict) else v
        usd = _build_usd_cfg(path=o.usd_path, scale=o.scale, body="dynamic", spawn=spawn)
        return RigidObject(RigidObjectCfg(prim_path=prim, spawn=usd,
                                          init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot)))

    # --------------------------------------------------------------- indices
    def _init_robot_indices(self):
        names = list(self.robot.joint_names)
        self.joint_names = names
        dev = self.device
        arm_all = robots.ARM["right"] + robots.ARM["left"]
        self.arm_ids = {s: torch.tensor([names.index(j) for j in robots.ARM[s]], device=dev) for s in robots.SIDES}
        self.hand_joints = {s: [j for j in names if j not in arm_all and self.spec.side_of(j) == s]
                            for s in robots.SIDES}
        self.actuated = {s: self.spec.actuated(s, self.hand_joints[s]) for s in robots.SIDES}
        bodies = list(self.robot.body_names)
        self.tip_ids, offs = [], []
        for s in robots.SIDES:
            for b, off in self.spec.tips[s]:
                assert b in bodies, f"{self.spec.name}: tip body {b} not in articulation"
                self.tip_ids.append(bodies.index(b))
                offs.append(off)
        self.tip_ids = torch.tensor(self.tip_ids, device=dev)
        self.tip_offset = torch.tensor(offs, dtype=torch.float32, device=dev)        # (10,3)
        self.wrist_ids = torch.tensor([bodies.index(self.spec.wrist[s]) for s in robots.SIDES], device=dev)
        self.flange_ids = torch.tensor([bodies.index(robots.FLANGE[s]) for s in robots.SIDES], device=dev)
        lim = self.robot.data.soft_joint_pos_limits[0]
        self.q_lo, self.q_hi = lim[:, 0].clone(), lim[:, 1].clone()

    def _init_object_bodies(self):
        """Flat list of object bodies: one per rigid object, every link of an articulation."""
        self.bodies = []                        # (obj_id, body_name, index in the asset)
        for k, a in self.objects.items():
            if isinstance(a, Articulation):
                self.bodies += [(k, b, i) for i, b in enumerate(a.body_names)]
            else:
                self.bodies.append((k, k, 0))
        self.obj_ids = list(self.objects)
        self.art_ids = [k for k, a in self.objects.items() if isinstance(a, Articulation)]

    # ------------------------------------------------------------------ state
    def tips(self):
        """(N,10,3) fingertip points, right hand then left, thumb..little."""
        p = self.robot.data.body_pos_w[:, self.tip_ids] - self.scene.env_origins[:, None]
        q = self.robot.data.body_quat_w[:, self.tip_ids]
        from isaaclab.utils.math import quat_apply
        return p + quat_apply(q.reshape(-1, 4), self.tip_offset.repeat(self.num_envs, 1)).reshape(p.shape)

    def body_pose(self, ids):
        o = self.scene.env_origins[:, None]
        return torch.cat([self.robot.data.body_pos_w[:, ids] - o, self.robot.data.body_quat_w[:, ids]], -1)

    def object_body_poses(self):
        """(N,B,7) poses of all object bodies."""
        o = self.scene.env_origins
        out = []
        for k, _, i in self.bodies:
            a = self.objects[k]
            out.append(torch.cat([a.data.body_pos_w[:, i] - o, a.data.body_quat_w[:, i]], -1))
        return torch.stack(out, 1) if out else torch.zeros(self.num_envs, 0, 7, device=self.device)

    def object_roots(self):
        """(N,n_obj,7) root poses and (N,n_obj,6) root velocities, in obj_ids order."""
        o = self.scene.env_origins[:, None]
        pos = torch.stack([a.data.root_pos_w for a in self.objects.values()], 1) - o
        quat = torch.stack([a.data.root_quat_w for a in self.objects.values()], 1)
        vel = torch.stack([torch.cat([a.data.root_lin_vel_w, a.data.root_ang_vel_w], -1)
                           for a in self.objects.values()], 1)
        return torch.cat([pos, quat], -1), vel

    def object_qpos(self):
        return {k: self.objects[k].data.joint_pos for k in self.art_ids}

    def teleport(self, q=None, obj_pose=None, obj_qpos=None, env_ids=None):
        """Write a kinematic state (zero velocity) and refresh every buffer, without stepping."""
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        n = len(ids)
        if q is not None:
            q = q.expand(n, -1).clone() if q.dim() == 1 else q
            self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
            self.robot.set_joint_position_target(q, env_ids=ids)
            self.q_target[ids] = q
        if obj_pose is not None:
            origin = self.scene.env_origins[ids]
            for j, a in enumerate(self.objects.values()):
                p = obj_pose[..., j, :].expand(n, -1).clone() if obj_pose.dim() == 2 else obj_pose[:, j].clone()
                p[:, :3] += origin
                a.write_root_pose_to_sim(p, env_ids=ids)
                a.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids=ids)
        for k, v in (obj_qpos or {}).items():
            v = v.expand(n, -1).clone() if v.dim() == 1 else v
            self.objects[k].write_joint_state_to_sim(v, torch.zeros_like(v), env_ids=ids)
            self.objects[k].set_joint_position_target(v, env_ids=ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=0.0)

    # ---------------------------------------------------------- success
    def success_states(self, i=0):
        """Object states of env i in the format Bench2Dex's success evaluators read."""
        roots, vel = self.object_roots()
        r, v = roots[i].cpu().numpy(), vel[i].cpu().numpy()
        out = {}
        for j, k in enumerate(self.obj_ids):
            st = {"pose_world": np.array([*r[j, :3], *r[j, 4:7], r[j, 3]], np.float32),
                  "lin_vel_world": v[j, :3], "ang_vel_world": v[j, 3:]}
            a = self.objects[k]
            if isinstance(a, Articulation):
                st["joint_names"] = list(a.joint_names)
                st["qpos"] = a.data.joint_pos[i].cpu().numpy()
                st["qvel"] = a.data.joint_vel[i].cpu().numpy()
            out[k] = st
        return out

    # --------------------------------------------------------- env plumbing
    def _pre_physics_step(self, actions):
        pass

    def _apply_action(self):
        self.robot.set_joint_position_target(self.q_target)
        for h in self.pre_step_hooks:
            h()

    def _get_observations(self):
        return {"policy": torch.zeros(self.num_envs, 1, device=self.device)}

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        self.frame += 1
        z = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return z, z

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)


def _scene_name(task: str) -> str:
    return paths.task_dir(task).parent.name


def to_t(a, device, dtype=torch.float32):
    return torch.as_tensor(np.asarray(a), dtype=dtype, device=device)
