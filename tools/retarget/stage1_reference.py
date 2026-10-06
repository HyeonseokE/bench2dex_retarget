"""Stage 1: the retargeting reference of one teleop episode, read off the simulator.

The source robot replays the episode kinematically (joint state, object root poses and
articulation joints written frame by frame, no physics step), so every quantity comes from the
same USD the benchmark simulates -- no URDF (Schunk's does not load, RH5DG2's lacks tip frames):

  body_pose      (T,B,7)  every object body: rigid objects, and each link of an articulation
  src_tips       (T,10,3) source fingertips, right then left, thumb..little
  src_wrist      (T,2,7)  source hand base;   src_flange (T,2,7) tool0
  contact_body   (T,10)   body each tip touches (-1: none), surface within 2 cm
  contact_local  (T,10,3) closest surface point, in that body's frame
  active         (T,2)    body each hand manipulates: moving, and within 3 cm of the hand

  python tools/retarget/stage1_reference.py --task 06 --episode 0 --headless
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--episode", type=int, required=True)
ap.add_argument("--chunk", type=int, default=256, help="frames teleported at once (one per env)")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # Bench2Dex root
from retarget import geometry, paths, robots  # noqa: E402
from retarget.sim_env import B2DEnv, B2DEnvCfg  # noqa: E402
from retarget.task import load_episode  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_conjugate  # noqa: E402

NEAR_DIST = 0.03     # a hand manipulates a moving body whose surface is this close to one of its tips
MOVING_SPEED = 0.03  # m/s, a body point moving faster than this counts as moving
DILATE = 5           # frames: grasp onset / release belong to the manipulation


def main():
    out_dir = paths.run_dir(args.task, args.episode)
    out = out_dir / "reference.npz"
    if out.exists():
        print(f"skip: {out} exists", flush=True)
        return
    ep = load_episode(args.task, args.episode)
    src = robots.get(ep.robot_key)
    T = ep.n
    cfg = B2DEnvCfg(task=args.task, episode=args.episode, robot=src.name)
    cfg.scene.num_envs = min(args.chunk, T)
    cfg.sim.device = args.device
    env = B2DEnv(cfg)
    env.reset()
    dev = env.device
    N = env.num_envs

    # ---- body meshes, in body frames, from the stage the simulator built
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    prim_paths = []
    for k, b, i in env.bodies:
        a = env.objects[k]
        prim_paths.append(a.root_physx_view.link_paths[0][i] if k in env.art_ids else a.root_physx_view.prim_paths[0])
    meshes = geometry.body_meshes(stage, prim_paths)
    for (k, b, _), m, pp in zip(env.bodies, meshes, prim_paths):
        print(f"  body {k}/{b}: {pp} -> {'no mesh' if m is None else f'{len(m[0])} verts {len(m[1])} faces'}", flush=True)

    # ---- kinematic replay
    perm = [ep.joint_names.index(n) for n in env.joint_names]
    q_src = torch.as_tensor(ep.qpos[:, perm], dtype=torch.float32, device=dev)
    roots = torch.as_tensor(np.stack([ep.obj_pose[k] for k in env.obj_ids], 1), dtype=torch.float32, device=dev)
    aq = {k: torch.as_tensor(ep.obj_qpos[k], dtype=torch.float32, device=dev) for k in env.art_ids}
    tips, wrist, flange, body = [], [], [], []
    for s in range(0, T, N):
        e = min(s + N, T)
        ids = torch.arange(e - s, device=dev)
        env.teleport(q=q_src[s:e], obj_pose=roots[s:e], obj_qpos={k: v[s:e] for k, v in aq.items()}, env_ids=ids)
        tips.append(env.tips()[: e - s])
        wrist.append(env.body_pose(env.wrist_ids)[: e - s])
        flange.append(env.body_pose(env.flange_ids)[: e - s])
        body.append(env.object_body_poses()[: e - s])
    tips, wrist, flange, body = (torch.cat(x) for x in (tips, wrist, flange, body))
    # sanity: the replayed roots must be where the episode put them
    rerr = (env.object_roots()[0][: e - s, :, :3] - roots[s:e, :, :3]).norm(dim=-1).max().item()
    print(f"replayed {T} frames; root position mismatch on the last chunk {100 * rerr:.3f} cm", flush=True)

    # ---- contacts: signed distance of every tip to every body surface
    B = len(env.bodies)
    dist = torch.full((T, 10, B), float("inf"), device=dev)
    local = torch.zeros(T, 10, B, 3, device=dev)
    for bi, m in enumerate(meshes):
        if m is None:
            continue
        V = torch.as_tensor(m[0], dtype=torch.float32, device=dev)
        Fc = torch.as_tensor(m[1], device=dev)
        p, qb = body[:, bi, :3], body[:, bi, 3:7]
        rel = (tips - p[:, None]).reshape(-1, 3)
        loc = quat_apply(quat_conjugate(qb).repeat_interleave(10, 0), rel)
        d, cpt = geometry.signed_distance(loc, V, Fc)
        dist[:, :, bi], local[:, :, bi] = d.reshape(T, 10), cpt.reshape(T, 10, 3)
    dist = geometry.effective_distance(dist)
    dmin, nearest = dist.min(-1)
    contact_body = torch.where(dmin < geometry.CONTACT_DIST, nearest, torch.full_like(nearest, -1))
    contact_local = torch.take_along_dim(local, nearest[..., None, None].expand(-1, -1, 1, 3), 2)[:, :, 0]

    # ---- active body per hand: moving (speed bound: origin speed + angular speed x body radius)
    radius = torch.tensor([0.0 if m is None else float(np.linalg.norm(m[0], axis=1).max()) for m in meshes],
                          device=dev)
    lin = torch.zeros(T, B, device=dev)
    lin[1:] = (body[1:, :, :3] - body[:-1, :, :3]).norm(dim=-1) * ep.fps
    rot = torch.zeros(T, B, device=dev)
    dq = (body[1:, :, 3:7] * body[:-1, :, 3:7]).sum(-1).abs().clamp(max=1.0)
    rot[1:] = 2.0 * torch.acos(dq) * ep.fps
    speed = lin + rot * radius                                       # upper bound of any surface point's speed
    moving = speed > MOVING_SPEED
    pad = torch.nn.functional.pad(moving.float().T[None], (DILATE, DILATE))[0]
    moving = torch.nn.functional.max_pool1d(pad[None], 2 * DILATE + 1, stride=1)[0].T > 0
    active = torch.full((T, 2), -1, dtype=torch.long, device=dev)
    for h in range(2):
        hd = dist[:, 5 * h:5 * h + 5].min(1).values                  # (T,B)
        cand = torch.where(moving & (hd < NEAR_DIST), hd, torch.full_like(hd, float("inf")))
        v, a = cand.min(1)
        active[:, h] = torch.where(torch.isfinite(v), a, torch.full_like(a, -1))
    act = active.cpu().numpy()
    for h in range(2):                                               # fill dropouts of <= 5 frames
        for t in range(1, T - 1):
            if act[t, h] == -1:
                prev, nxt = act[max(0, t - DILATE):t, h], act[t + 1:t + 1 + DILATE, h]
                if len(prev) and prev[-1] >= 0 and prev[-1] in nxt:
                    act[t, h] = prev[-1]

    names = [f"{k}/{b}" if k in env.art_ids else k for k, b, _ in env.bodies]
    segs = {}
    for h, side in enumerate(robots.SIDES):
        segs[side], start = [], None
        for t, a in enumerate(list(act[:, h]) + [-1]):
            if start is not None and a != act[start, h]:
                if t - start >= 10:
                    segs[side].append({"body": names[act[start, h]], "start": start, "end": t})
                start = None
            if start is None and a >= 0:
                start = t
    meta = {"task": paths.task_dir(args.task).parent.name, "episode": args.episode, "source": src.name,
            "n_frames": T, "fps": ep.fps, "obj_ids": env.obj_ids, "bodies": names,
            "body_obj": [env.obj_ids.index(k) for k, _, _ in env.bodies],
            "art": {k: list(env.objects[k].joint_names) for k in env.art_ids},
            "table_z": env.table_heights.table_z, "segments": segs, "source_episode": str(ep.path)}
    arrays = {"body_pose": body, "obj_root": roots, "src_tips": tips, "src_wrist": wrist, "src_flange": flange,
              "src_q": q_src, "contact_body": contact_body, "contact_local": contact_local, "contact_dist": dmin}
    arrays = {k: v.cpu().numpy() for k, v in arrays.items()}
    arrays["active"] = act
    for k in env.art_ids:
        arrays[f"art/{k}"] = aq[k].cpu().numpy()
    np.savez_compressed(out, meta=np.array(json.dumps(meta)), joint_names=np.array(env.joint_names), **arrays)
    (out_dir / "reference.json").write_text(json.dumps(meta, indent=1))    # readable without numpy
    np.savez_compressed(out_dir / "meshes.npz", **{f"{i}/v": m[0] for i, m in enumerate(meshes) if m is not None},
                        **{f"{i}/f": m[1] for i, m in enumerate(meshes) if m is not None})
    ct = (arrays["contact_body"] >= 0).mean(0)
    print(f"contact fraction per tip R {np.round(ct[:5], 2).tolist()} L {np.round(ct[5:], 2).tolist()}")
    print(json.dumps(segs, indent=1))
    print(f"RESULT stage1 {meta['task']} ep{args.episode}: {T} frames, {B} bodies -> {out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
