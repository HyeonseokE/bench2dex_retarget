"""Stage 2: kinematic retargeting, source fingertips -> target robot (whole arm + hand IK).

Same objective as the task-06 prototype (retarget/retarget_kinematic.py): both robots stand in the
same world over the same objects, so target fingertip k goes to source fingertip k; tips in contact
weigh 10, free tips 2; the tool0 flange is softly kept on the source flange pose; a small pull
keeps each frame near the previous one, and frames are solved in order, each warm-started from the
last (solving all frames in parallel let neighbouring frames settle on different finger branches:
1.1 rad jumps on Shadow's little finger, task 06 ep0). What changed is where the kinematics come
from: the target robot's own USD in the simulator, not a URDF.

  * FK and Jacobians are read from PhysX after a zero-gravity step on the commanded joints, so
    the Jacobian always belongs to the configuration it is used at.
  * Variables are the arm joints and the hand joints teleop actuates; the others follow the
    coupling fitted on the hand's own demos (retarget.coupling), so the IK only plans finger shapes
    the hand can take.

  python tools/retarget/stage2_kinematic.py --task 06 --episode 0 --target shadow --headless
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--episode", type=int, required=True)
ap.add_argument("--target", required=True)
ap.add_argument("--iters0", type=int, default=200,
                help="iterations on the first frame: converge fully, or frame 1 inherits the remainder as a jump")
ap.add_argument("--iters", type=int, default=8, help="iterations per later frame (warm-started)")
ap.add_argument("--flange_w", type=float, default=-1.0,
                help="scale of the soft flange-pose term; -1: 1 when source and target share the arm, else 0 (another "
                     "arm's flange frame has its own axis convention, so its pose is not a target for this arm's flange)")
ap.add_argument("--starts", type=int, default=-1,
                help="first-frame IK restarts (random arm poses), best fingertip fit kept; -1: 32 when source and "
                     "target arms differ (one start from the target's home pose lands in a wrong arm branch: "
                     "task 80 JAKA -> UR5, 5-7 cm), else 1")
ap.add_argument("--wrap", type=int, default=1,
                help="1: every arm joint takes the 2*pi-equivalent angle nearest the previous frame (within its limits). "
                     "UR5 joints span +-2*pi, so the IK could land on an equivalent angle 4*pi away (43 -> rh56dfx: "
                     "12.57 rad in one frame). 0 = original behaviour")
ap.add_argument("--arm_step", type=float, default=0.0,
                help="rad per frame (20 fps) an arm joint may move between frames; 0 = no limit (the original "
                     "behaviour). Without it the IK hopped between wrist branches (0.5-1.5 rad/frame on Schunk/RH56DFX) "
                     "and to 4*pi-equivalent angles (43 -> rh56dfx), which SPIDER's smoothed arm could not follow")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # Bench2Dex root
from retarget import coupling, paths, robots  # noqa: E402
from retarget.sim_env import B2DEnv, B2DEnvCfg  # noqa: E402
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul  # noqa: E402

W_TIP_CONTACT = 10.0
W_TIP_FREE = 2.0
W_FLANGE_ROT = 0.1
W_FLANGE_POS = 0.05
W_REG = 0.05
DAMPING = 1e-3


def skew(v):
    z = torch.zeros_like(v[..., 0])
    return torch.stack([torch.stack([z, -v[..., 2], v[..., 1]], -1),
                        torch.stack([v[..., 2], z, -v[..., 0]], -1),
                        torch.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def main():
    rdir = paths.run_dir(args.task, args.episode)
    out_dir = rdir / args.target
    out = out_dir / "kinematic.npz"
    if out.exists():
        print(f"skip: {out} exists", flush=True)
        return
    ref = np.load(rdir / "reference.npz", allow_pickle=True)
    meta = json.loads(str(ref["meta"]))
    T = meta["n_frames"]
    cfg = B2DEnvCfg(task=args.task, episode=args.episode, robot=args.target, with_objects=False, gravity=False)
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    env = B2DEnv(cfg)
    env.reset()
    dev, names = env.device, env.joint_names
    spec = env.spec

    rules = coupling.fit(spec, names, env.hand_joints, env.actuated)["rules"]
    free = spec.arm["right"] + spec.arm["left"] + env.actuated["right"] + env.actuated["left"]
    C, d = coupling.matrix(names, free, rules, spec.locked)
    C, d = torch.as_tensor(C, dtype=torch.float32, device=dev), torch.as_tensor(d, dtype=torch.float32, device=dev)
    free_ids = torch.tensor([names.index(n) for n in free], device=dev)
    x_lo, x_hi = env.q_lo[free_ids], env.q_hi[free_ids]

    n_arm = len(spec.arm["right"]) + len(spec.arm["left"])          # free[:n_arm] are the arm joints
    src_names = [str(n) for n in ref["joint_names"]]
    same_arm = all(n in src_names for n in free[:n_arm])
    fw = args.flange_w if args.flange_w >= 0 else (1.0 if same_arm else 0.0)
    # first frame's initial guess: hands at home; arm = the source arm configuration when source and
    # target share the arm (same joint names, e.g. the five UR5 robots), else the target's home pose
    # (the longer first-frame solve, --iters0, then moves it onto the source fingertips)
    src_names = [str(n) for n in ref["joint_names"]]
    x0 = env.robot.data.default_joint_pos[0, free_ids].clone()
    if all(n in src_names for n in free[:n_arm]):
        for i, n in enumerate(free[:n_arm]):
            x0[i] = float(ref["src_q"][0, src_names.index(n)])
    x0 = torch.max(torch.min(x0, x_hi), x_lo)

    tgt_tips = torch.as_tensor(ref["src_tips"], dtype=torch.float32, device=dev)              # (T,10,3)
    w_tip = torch.where(torch.as_tensor(ref["contact_body"], device=dev) >= 0, W_TIP_CONTACT, W_TIP_FREE)
    w_tip = w_tip * env.tip_w                                   # 0 for a finger the target hand lacks (LEAP)
    fl = torch.as_tensor(ref["src_flange"], dtype=torch.float32, device=dev)                  # (T,2,7)
    fixed = env.robot.is_fixed_base
    assert fixed, "robot bases are fixed; the Jacobian indexing below assumes it"

    two_pi = 2 * torch.pi

    def wrap_arm(x, x_prev):
        """Arm joints moved to the 2*pi-equivalent angle closest to x_prev, if that one is within limits."""
        a, p = x[0, :n_arm], x_prev[:n_arm]
        near = p + torch.remainder(a - p + torch.pi, two_pi) - torch.pi
        ok = (near >= x_lo[:n_arm]) & (near <= x_hi[:n_arm])
        x = x.clone()
        x[0, :n_arm] = torch.where(ok, near, a)
        return x

    def solve(t, x, x_reg, iters, x_prev=None):
        """Frame t, starting from x (n_free,), regularised towards x_reg; arm joints kept within
        args.arm_step of the previous frame's solution x_prev (if given)."""
        x = x[None]
        lo, hi = x_lo.clone(), x_hi.clone()
        if x_prev is not None and args.arm_step > 0:
            lo[:n_arm] = torch.max(lo[:n_arm], x_prev[:n_arm] - args.arm_step)
            hi[:n_arm] = torch.min(hi[:n_arm], x_prev[:n_arm] + args.arm_step)
        for _ in range(iters):
            q = torch.max(torch.min(x @ C.T + d, env.q_hi), env.q_lo)
            env.teleport(q=q)
            env.sim.step(render=False)                      # zero gravity, targets = q: stays put
            env.scene.update(dt=env.physics_dt)
            J = env.robot.root_physx_view.get_jacobians()   # (1, bodies-1, 6, dofs)
            tips = env.tips()
            tb = env.robot.data.body_pos_w[:, env.tip_ids] - env.scene.env_origins[:, None]
            Jt = J[:, env.tip_ids - 1]                      # (1,10,6,J)
            Jtip = Jt[:, :, :3] - skew(tips - tb) @ Jt[:, :, 3:]
            fpose = env.body_pose(env.flange_ids)
            Jf = J[:, env.flange_ids - 1]                   # (1,2,6,J)
            rot = axis_angle_from_quat(quat_mul(fl[t, :, 3:7], quat_conjugate(fpose[0, :, 3:7])))[None]
            e = torch.cat([(w_tip[t, :, None] * (tgt_tips[t] - tips)).reshape(1, -1),
                           (fw * W_FLANGE_POS * (fl[t, :, :3] - fpose[..., :3])).reshape(1, -1),
                           (fw * W_FLANGE_ROT * rot).reshape(1, -1),
                           W_REG * (x_reg[None] - x)], -1)
            Jx = torch.cat([(w_tip[t, :, None, None] * Jtip).reshape(1, -1, len(names)) @ C,
                            (fw * W_FLANGE_POS * Jf[:, :, :3]).reshape(1, -1, len(names)) @ C,
                            (fw * W_FLANGE_ROT * Jf[:, :, 3:]).reshape(1, -1, len(names)) @ C,
                            W_REG * torch.eye(len(free), device=dev)[None]], 1)
            H = Jx.transpose(1, 2) @ Jx + DAMPING * torch.eye(len(free), device=dev)
            dx = torch.linalg.solve(H, (Jx.transpose(1, 2) @ e[..., None]))[..., 0]
            # original (before 2026-10-08), joint limits only:
            # x = torch.max(torch.min(x + dx, x_hi), x_lo)
            x = torch.max(torch.min(x + dx, hi), lo)
            if x_prev is not None and args.wrap:
                x = wrap_arm(x, x_prev)
        return x[0]

    n_starts = args.starts if args.starts > 0 else (1 if same_arm else 32)

    def frame0_err(x):
        q = torch.max(torch.min(x[None] @ C.T + d, env.q_hi), env.q_lo)
        env.teleport(q=q)
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        return float((w_tip[0, :, None] * (tgt_tips[0] - env.tips()[0])).norm(dim=-1).sum())

    best = None
    for k in range(n_starts):
        xk = x0.clone()
        if k > 0:
            xk[:n_arm] = x_lo[:n_arm] + (x_hi[:n_arm] - x_lo[:n_arm]) * torch.rand(n_arm, device=dev)
        sol = solve(0, xk, xk, args.iters0)
        err = frame0_err(sol)
        if best is None or err < best[0]:
            best = (err, k, sol)
    if n_starts > 1:
        print(f"  first frame: best of {n_starts} starts = start {best[1]} (weighted tip error {best[0]:.3f})", flush=True)
    xs = [best[2]]
    for t in range(1, T):
        # original (before 2026-10-08), no per-frame arm limit:
        # xs.append(solve(t, xs[-1], xs[-1], args.iters))
        xs.append(solve(t, xs[-1], xs[-1], args.iters, x_prev=xs[-1]))   # x_prev: wrap / arm_step
        if t % 100 == 0:
            print(f"  frame {t}/{T}", flush=True)
    x = torch.stack(xs)

    # FK of the solution, all frames at once
    tips_all, wrist_all, flange_all = [], [], []
    q_all = torch.max(torch.min(x @ C.T + d, env.q_hi), env.q_lo)
    for t in range(T):
        env.teleport(q=q_all[t:t + 1])
        tips_all.append(env.tips()[0])
        wrist_all.append(env.body_pose(env.wrist_ids)[0])
        flange_all.append(env.body_pose(env.flange_ids)[0])
    q = q_all
    tips = torch.stack(tips_all)
    err = (tips - tgt_tips).norm(dim=-1)
    contact = w_tip == W_TIP_CONTACT
    rep = {"tip_err_cm_mean": round(100 * err.mean().item(), 2),
           "tip_err_cm_contact": round(100 * err[contact].mean().item(), 2) if contact.any() else None,
           "tip_err_cm_p95": round(100 * err.quantile(0.95).item(), 2),
           "q_jump_max_rad": round((q[1:] - q[:-1]).abs().max().item(), 3),
           "arm_step_max_rad": round((x[1:, :n_arm] - x[:-1, :n_arm]).abs().max().item(), 3),
           "arm_step_limit_rad": args.arm_step, "wrap": args.wrap}
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, q=q.cpu().numpy(), joint_names=np.array(names), tips=tips.cpu().numpy(),
                        wrist=torch.stack(wrist_all).cpu().numpy(),
                        flange=torch.stack(flange_all).cpu().numpy(), tip_err=err.cpu().numpy(),
                        free=np.array(free), report=np.array(json.dumps(rep)))
    print(f"RESULT stage2 {meta['task']} ep{args.episode} -> {args.target}: {rep}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
