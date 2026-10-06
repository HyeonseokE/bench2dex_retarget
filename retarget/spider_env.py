"""The SPIDER rollout environment: target robot, objects, the retargeted reference.

Control, as in the task-06 prototype (retarget/dexmachina_env.py, the settings SPIDER v2 used):
  wrist    a 6-DoF residual pose per hand on the reference wrist (hand base) pose, scaled by
           wrist_trans_scale / wrist_rot_scale and realised by one damped-least-squares IK step
           on that arm, integrated on the previous command (ik_from_command).
  fingers  a residual on the actuated hand joints, +-finger_res_cap rad around the reference;
           the non-actuated joints follow retarget.coupling, locked ones stay at 0.
  gate     residuals are faded in only around the frames a hand manipulates something
           (gate_margin, gate_ramp); elsewhere the robot replays the reference.
  ema      joint targets are low-passed: target = ema * previous + (1 - ema) * new.
Gravity compensation is whatever the Bench2Dex spawner installed (pre_step_hooks).

Object-side guidance is SPIDER's virtual contact spring: each demo contact point on a body is
pulled towards the fingertip that touched it, with light damping and capped at a few body
weights. It works on articulation links as well as rigid objects. SPIDER anneals its gain to 0
over the optimisation iterations and committed frames always run with gain 0.

Before optimising, ``hold_pass`` replays the kinematic reference with the robot under PD control
while every object is held on its demo trajectory (DexMachina's post-processing): the joint
values the robot actually reaches are free of hand-object penetration and become the reference.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.utils.math import axis_angle_from_quat, quat_apply, quat_conjugate, quat_from_angle_axis, quat_mul

from isaaclab.utils import configclass

from . import coupling, paths, robots
from .sim_env import B2DEnv, B2DEnvCfg

# object rotation error by shape: "none" for round things, "tilt" (only the up axis) for things
# symmetric about it; matched against the asset key
ROT_NONE = ("apple", "orange", "lemon", "ball", "golf", "tennis", "table_tennis")
ROT_TILT = ("bowl", "plate", "wineglass", "cup", "pot", "basket", "bottle", "soy", "vinegar", "olive")


@configclass
class SpiderEnvCfg(B2DEnvCfg):
    gate_margin: int = 15
    gate_ramp: int = 5
    action_ema: float = 0.8
    wrist_trans_scale: float = 0.05
    wrist_rot_scale: float = 0.5
    finger_res_cap: float = 0.6        # SPIDER v2 on task 06 (0.25 crushed thin objects less, but grasped worse)
    ik_damping: float = 0.05


class SpiderEnv(B2DEnv):
    cfg: SpiderEnvCfg

    def __init__(self, cfg: SpiderEnvCfg, **kw):
        super().__init__(cfg, **kw)
        dev = self.device
        rdir = paths.run_dir(cfg.task, cfg.episode)
        ref = np.load(rdir / "reference.npz", allow_pickle=True)
        kin = np.load(rdir / cfg.robot / "kinematic.npz", allow_pickle=True)
        self.meta = json.loads(str(ref["meta"]))
        assert self.meta["obj_ids"] == self.obj_ids and len(self.meta["bodies"]) == len(self.bodies)
        self.T = self.meta["n_frames"]
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)
        perm = [list(kin["joint_names"]).index(n) for n in self.joint_names]
        self.ref_q = t(kin["q"][:, perm])
        self.ref_tips = t(kin["tips"])
        self.ref_wrist = t(kin["wrist"])
        self.ref_body = t(ref["body_pose"])                                   # (T,B,7)
        self.ref_root = t(ref["obj_root"])                                    # (T,n_obj,7)
        self.ref_art = {k: t(ref[f"art/{k}"]) for k in self.art_ids}
        self.ref_root_vel = torch.zeros_like(self.ref_root[..., :6])
        self.ref_root_vel[1:-1, :, :3] = (self.ref_root[2:, :, :3] - self.ref_root[:-2, :, :3]) * (0.5 / self.step_dt)
        self.contact_body = torch.as_tensor(ref["contact_body"], dtype=torch.long, device=dev)   # (T,10)
        self.contact_local = t(ref["contact_local"])
        self.active = np.asarray(ref["active"])                               # (T,2) body index
        self.body_obj = torch.as_tensor(self.meta["body_obj"], device=dev)    # body -> object index
        self.gate = t(self._make_gate(self.active, cfg.gate_margin, cfg.gate_ramp))   # (T,2)

        # controlled joints and their coupling
        rules = coupling.fit(self.spec, self.joint_names, self.hand_joints, self.actuated)["rules"]
        self.act_names = self.actuated["right"] + self.actuated["left"]
        self.act_ids = torch.tensor([self.joint_names.index(n) for n in self.act_names], device=dev)
        self.act_side = torch.tensor([0] * len(self.actuated["right"]) + [1] * len(self.actuated["left"]), device=dev)
        pas = [(self.joint_names.index(j), self.joint_names.index(r["master"]) if r["master"] else -1, r["a"], r["b"])
               for j, r in rules.items()]
        self.pas_ids = torch.tensor([p[0] for p in pas], dtype=torch.long, device=dev)
        self.pas_master = torch.tensor([p[1] for p in pas], dtype=torch.long, device=dev)
        self.pas_a, self.pas_b = t([p[2] for p in pas]), t([p[3] for p in pas])
        self.locked_ids = torch.tensor([self.joint_names.index(j) for j in self.spec.locked if j in self.joint_names],
                                       dtype=torch.long, device=dev)
        self.n_act = 12 + len(self.act_names)
        self.actions = torch.zeros(self.num_envs, self.n_act, device=dev)
        self.prev_actions = self.actions.clone()

        # body masses / inertias for the guidance spring
        n = self.num_envs
        m, inert = [], []
        for k, _, i in self.bodies:
            v = self.objects[k].root_physx_view
            mm, ii = v.get_masses().reshape(n, -1), v.get_inertias().reshape(n, -1, 9)
            m.append(mm[:, i])
            inert.append(ii[:, i][:, [0, 4, 8]].mean(-1))
        self.body_mass = torch.stack(m, 1)[..., None].to(dev)                 # (N,B,1)
        self.body_inertia = torch.stack(inert, 1)[..., None].to(dev)
        self.asset_bodies = {k: [j for j, (kk, _, _) in enumerate(self.bodies) if kk == k] for k in self.obj_ids}
        codes = [self.scene_def.objects[[o.obj_id for o in self.scene_def.objects].index(k)].asset_key
                 for k in self.obj_ids]
        mode = ["none" if any(s in c for s in ROT_NONE) else "tilt" if any(s in c for s in ROT_TILT) else "full"
                for c in codes]
        self.rot_full = torch.tensor([x == "full" for x in mode], device=dev)
        self.rot_tilt = torch.tensor([x == "tilt" for x in mode], device=dev)
        self.guide_gain = 0.0
        self.hold = False

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _make_gate(active, margin, ramp):
        T, H = active.shape
        g = np.zeros((T, H), np.float32)
        for h in range(H):
            idx = np.where(active[:, h] >= 0)[0]
            if len(idx):
                dd = np.abs(np.arange(T)[:, None] - idx[None]).min(1)
                g[:, h] = np.clip(1.0 - (dd - margin) / max(ramp, 1), 0.0, 1.0)
        return g

    def f(self, offset=0):
        return torch.clamp(self.frame + offset, max=self.T - 1)

    def obj_rot_dist(self, quat, ref):
        full = axis_angle_from_quat(quat_mul(ref.reshape(-1, 4), quat_conjugate(quat.reshape(-1, 4)))).norm(dim=-1)
        full = full.reshape(quat.shape[:-1])
        z = torch.zeros_like(quat[..., 1:])
        z[..., 2] = 1.0
        a = quat_apply(quat.reshape(-1, 4), z.reshape(-1, 3)).reshape(z.shape)
        b = quat_apply(ref.reshape(-1, 4), z.reshape(-1, 3)).reshape(z.shape)
        tilt = torch.acos((a * b).sum(-1).clamp(-1.0, 1.0))
        return torch.where(self.rot_full, full, torch.where(self.rot_tilt, tilt, torch.zeros_like(full)))

    def demo_contact_world(self, body_pose, f):
        """Demo contact points of frame f placed on the current body poses: (N,10,3), valid (N,10)."""
        cb = self.contact_body[f]
        valid = cb >= 0
        idx = cb.clamp(min=0)
        p = torch.gather(body_pose[..., :3], 1, idx[..., None].expand(-1, -1, 3))
        q = torch.gather(body_pose[..., 3:7], 1, idx[..., None].expand(-1, -1, 4))
        loc = self.contact_local[f]
        return quat_apply(q.reshape(-1, 4), loc.reshape(-1, 3)).reshape(p.shape) + p, valid, idx

    # ------------------------------------------------------------------ step
    def _pre_physics_step(self, actions):
        cfg = self.cfg
        self.prev_actions = self.actions.clone()
        self.actions = actions.clamp(-1.0, 1.0)
        q_prev = self.q_target.clone()
        f1 = self.f(1)
        if self.hold:                      # hold pass: track the kinematic reference exactly
            self.q_target = self.ref_q[f1].clone()
            return
        q = self.ref_q[f1].clone()
        gate = self.gate[f1]                                                   # (N,2)
        # fingers: residual on actuated joints, coupled ones follow
        a_h = self.actions[:, 12:] * gate[:, self.act_side]
        q[:, self.act_ids] = q[:, self.act_ids] + cfg.finger_res_cap * a_h
        q = torch.max(torch.min(q, self.q_hi), self.q_lo)
        if len(self.pas_ids):
            master = q[:, self.pas_master.clamp(min=0)]
            q[:, self.pas_ids] = torch.where(self.pas_master >= 0, self.pas_a * master + self.pas_b, self.pas_b)
        if len(self.locked_ids):
            q[:, self.locked_ids] = 0.0
        # wrists: residual pose on the reference, one DLS-IK step per arm
        wp = self.body_pose(self.wrist_ids)
        ref_w = self.ref_wrist[f1]                                             # (N,2,7)
        jac = self.robot.root_physx_view.get_jacobians()
        for s in range(2):
            a = self.actions[:, 6 * s:6 * s + 6] * gate[:, s:s + 1]
            tgt_pos = ref_w[:, s, :3] + cfg.wrist_trans_scale * a[:, :3]
            rv = cfg.wrist_rot_scale * a[:, 3:]
            ang = rv.norm(dim=-1)
            dq_rot = quat_from_angle_axis(ang, rv / ang.clamp(min=1e-6)[:, None])
            tgt_quat = quat_mul(dq_rot, ref_w[:, s, 3:7])
            err = torch.cat([tgt_pos - wp[:, s, :3],
                             axis_angle_from_quat(quat_mul(tgt_quat, quat_conjugate(wp[:, s, 3:7])))], -1)
            J = jac[:, self.wrist_ids[s] - 1][:, :, self.arm_ids[robots.SIDES[s]]]
            JJt = J @ J.transpose(1, 2) + (cfg.ik_damping ** 2) * torch.eye(6, device=self.device)
            dq = (J.transpose(1, 2) @ torch.linalg.solve(JJt, err[..., None]))[..., 0]
            ai = self.arm_ids[robots.SIDES[s]]
            q[:, ai] = q_prev[:, ai] + dq
        q = torch.max(torch.min(q, self.q_hi), self.q_lo)
        self.q_target = cfg.action_ema * q_prev + (1.0 - cfg.action_ema) * q

    def commit_step(self, action=None, after_physics_step=None):
        """One control step of the committed rollout, like DirectRLEnv.step but with a hook after
        every physics step (Bench2Dex updates its metrics there). ``action=None`` holds the current
        joint targets (settle). Returns env 0's executed joint target."""
        if action is not None:
            self._pre_physics_step(action[None].expand(self.num_envs, -1))
        cmd = self.q_target[0].detach().cpu().numpy().copy()
        for _ in range(self.cfg.decimation):
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
            if after_physics_step is not None:
                after_physics_step()
        if action is not None:
            self.frame += 1
        return cmd

    def _apply_action(self):
        super()._apply_action()                       # joint targets + gravity compensation hooks
        if self.hold:
            f1 = self.f(1)
            self._write_objects(self.ref_root[f1], {k: v[f1] for k, v in self.ref_art.items()})
            return
        self._apply_guidance()

    def _apply_guidance(self):
        n, dev = self.num_envs, self.device
        B = len(self.bodies)
        F = torch.zeros(n, B, 3, device=dev)
        Tq = torch.zeros(n, B, 3, device=dev)
        if self.guide_gain > 0.0:
            body = self.object_body_poses()
            cp, valid, idx = self.demo_contact_world(body, self.f(1))
            fi = self.guide_gain * (self.tips() - cp) * valid[..., None].float()
            bpos = torch.gather(body[..., :3], 1, idx[..., None].expand(-1, -1, 3))
            ti = torch.cross(cp - bpos, fi, dim=-1)
            F.scatter_add_(1, idx[..., None].expand(-1, -1, 3), fi)
            Tq.scatter_add_(1, idx[..., None].expand(-1, -1, 3), ti)
            held = torch.zeros(n, B, 1, device=dev).scatter_add_(1, idx[..., None], valid[..., None].float()) > 0
            lin = self._body_lin_vel()
            F = F - 2.0 * self.body_mass * lin * held                            # light damping on guided bodies
            cap = 3.0 * 9.81 * self.body_mass
            F = F * torch.clamp(cap / F.norm(dim=-1, keepdim=True).clamp(min=1e-6), max=1.0)
            tcap = 80.0 * self.body_inertia
            Tq = Tq * torch.clamp(tcap / Tq.norm(dim=-1, keepdim=True).clamp(min=1e-9), max=1.0)
        for k, ids in self.asset_bodies.items():
            a = self.objects[k]
            a.permanent_wrench_composer.set_forces_and_torques(
                forces=F[:, ids].contiguous(), torques=Tq[:, ids].contiguous(),
                body_ids=[self.bodies[j][2] for j in ids] if isinstance(a, Articulation) else None, is_global=True)

    def _body_lin_vel(self):
        out = []
        for k, _, i in self.bodies:
            out.append(self.objects[k].data.body_lin_vel_w[:, i])
        return torch.stack(out, 1)

    def _write_objects(self, roots, art):
        n = self.num_envs
        origin = self.scene.env_origins
        zero = torch.zeros(n, 6, device=self.device)
        for j, (k, a) in enumerate(self.objects.items()):
            p = roots[:, j].clone()
            p[:, :3] += origin
            a.write_root_pose_to_sim(p)
            a.write_root_velocity_to_sim(zero)
            if k in art:
                a.write_joint_state_to_sim(art[k], torch.zeros_like(art[k]))

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not hasattr(self, "ref_q"):
            return
        self.frame[env_ids] = 0
        q = self.ref_q[0].expand(len(env_ids), -1).clone()
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        self.robot.set_joint_position_target(q, env_ids=env_ids)
        self.q_target[env_ids] = q
        origin = self.scene.env_origins[env_ids]
        for j, (k, a) in enumerate(self.objects.items()):
            p = self.ref_root[0, j].expand(len(env_ids), -1).clone()
            p[:, :3] += origin
            a.write_root_pose_to_sim(p, env_ids=env_ids)
            a.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids)
            if k in self.ref_art:
                v = self.ref_art[k][0].expand(len(env_ids), -1).clone()
                a.write_joint_state_to_sim(v, torch.zeros_like(v), env_ids=env_ids)
                a.set_joint_position_target(v, env_ids=env_ids)
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0

    # --------------------------------------------------------- snapshots
    def snapshot(self, i=0):
        s = {"q": self.robot.data.joint_pos[i].clone(), "qd": self.robot.data.joint_vel[i].clone(),
             "q_target": self.q_target[i].clone(), "actions": self.actions[i].clone(), "frame": int(self.frame[i]),
             "obj": {}}
        o = self.scene.env_origins[i]
        for k, a in self.objects.items():
            d = {"pose": torch.cat([a.data.root_pos_w[i] - o, a.data.root_quat_w[i]]).clone(),
                 "vel": torch.cat([a.data.root_lin_vel_w[i], a.data.root_ang_vel_w[i]]).clone()}
            if isinstance(a, Articulation):
                d["jp"], d["jv"] = a.data.joint_pos[i].clone(), a.data.joint_vel[i].clone()
                d["jt"] = a.data.joint_pos_target[i].clone()
            s["obj"][k] = d
        return s

    def restore(self, s):
        n = self.num_envs
        ex = lambda v: v.expand(n, *v.shape).clone()
        self.robot.write_joint_state_to_sim(ex(s["q"]), ex(s["qd"]))
        self.robot.set_joint_position_target(ex(s["q_target"]))
        self.q_target[:] = s["q_target"]
        self.actions[:] = s["actions"]
        self.prev_actions = self.actions.clone()
        for k, a in self.objects.items():
            d = s["obj"][k]
            p = ex(d["pose"])
            p[:, :3] += self.scene.env_origins
            a.write_root_pose_to_sim(p)
            a.write_root_velocity_to_sim(ex(d["vel"]))
            if "jp" in d:
                a.write_joint_state_to_sim(ex(d["jp"]), ex(d["jv"]))
                a.set_joint_position_target(ex(d["jt"]))
        self.frame[:] = s["frame"]
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=0.0)


class SuccessTracker:
    """Bench2Dex success, evaluated every control step on env 0, with Bench2Dex's stable-success
    rule: the whole condition holding for dwell_s, latched. The evaluator context carries state
    (``sequence`` progress, ``hold_duration`` timers), so it is part of every snapshot."""

    def __init__(self, scene, step_dt):
        paths.use_bench2dex()
        from success import evaluate_success_conditions  # Bench2Dex

        self._eval = evaluate_success_conditions
        self.conditions = scene.success_conditions
        self.dwell = int(np.ceil(scene.dwell_s / step_dt - 1e-9))
        self.ctx = {"dt": float(step_dt)}
        self.run = 0
        self.history = []

    def update(self, states) -> bool:
        ok = bool(self._eval(self.conditions, states, self.ctx)) if self.conditions else False
        self.run = self.run + 1 if ok else 0
        self.history.append(ok)
        return ok

    def stable_frame(self):
        run = 0
        for i, ok in enumerate(self.history):
            run = run + 1 if ok else 0
            if run >= self.dwell:
                return i
        return None

    def state(self):
        return copy.deepcopy((self.ctx, self.run, len(self.history)))

    def load(self, st):
        self.ctx, self.run, n = copy.deepcopy(st)
        del self.history[n:]


def make_cfg(task, episode, robot, num_envs, device, **overrides) -> SpiderEnvCfg:
    cfg = SpiderEnvCfg(task=task, episode=episode, robot=robots.get(robot).name)
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
