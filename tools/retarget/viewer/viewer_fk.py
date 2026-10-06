"""Web-viewer helper (Isaac): visual meshes and per-frame link poses of one robot.

Spawns the robot alone (Bench2Dex spawner, no objects, no gravity), reads each link's visual meshes
in the link frame (collision meshes skipped, scale baked in), then teleports through the joint
trajectories given in --q and records every link's world pose. One process per robot, because an
Isaac Lab scene cannot be rebuilt in one process.

  python tools/retarget/viewer/viewer_fk.py --task 06 --episode 0 --robot shadow \
      --q in.npz --out fk.npz --headless

in.npz:  joint_names (J,), and any number of trajectories "traj/<name>" (T,J) in that joint order
fk.npz:  link_names (L,), mesh/<i>/v (n,3) float32 and mesh/<i>/f (m,3) int32 per link (missing:
         no visual), pose/<name> (T,L,7) as x y z qx qy qz qw (three.js order)
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--episode", type=int, required=True)
ap.add_argument("--robot", required=True)
ap.add_argument("--q", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--chunk", type=int, default=128)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from retarget.sim_env import B2DEnv, B2DEnvCfg  # noqa: E402


def visual_meshes(stage, link_paths):
    """Per link: (verts, faces) of its visual meshes in the link frame, or None."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    cache = UsdGeom.XformCache()
    out = []
    for path in link_paths:
        body = stage.GetPrimAtPath(path)
        T = Gf.Transform(cache.GetLocalToWorldTransform(body))
        inv = Gf.Matrix4d(T.GetRotation(), T.GetTranslation()).GetInverse()
        verts, faces, off = [], [], 0
        it = iter(Usd.PrimRange(body, Usd.TraverseInstanceProxies()))
        for p in it:
            if p != body and p.HasAPI(UsdPhysics.RigidBodyAPI):
                it.PruneChildren()
                continue
            if "collision" in p.GetName().lower():
                it.PruneChildren()
                continue
            img = UsdGeom.Imageable(p)
            if img and img.ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
                continue
            if img and img.ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue
            if not p.IsA(UsdGeom.Mesh):
                continue
            m = UsdGeom.Mesh(p)
            pts, counts, idx = (m.GetPointsAttr().Get(), m.GetFaceVertexCountsAttr().Get(),
                                m.GetFaceVertexIndicesAttr().Get())
            if not pts or not counts or not idx:
                continue
            M = np.array(cache.GetLocalToWorldTransform(p) * inv)
            P = np.asarray(pts, dtype=np.float64) @ M[:3, :3] + M[3, :3]
            idx, counts = np.asarray(idx), np.asarray(counts)
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            tris = [(idx[s], idx[s + j], idx[s + j + 1]) for s, c in zip(starts, counts) for j in range(1, c - 1)]
            verts.append(P.astype(np.float32))
            faces.append(np.asarray(tris, dtype=np.int64) + off)
            off += len(P)
        out.append((np.concatenate(verts), np.concatenate(faces).astype(np.int32)) if verts else None)
    return out


def main():
    data = np.load(args.q)
    cfg = B2DEnvCfg(task=args.task, episode=args.episode, robot=args.robot, with_objects=False, gravity=False)
    cfg.scene.num_envs = args.chunk
    cfg.sim.device = args.device
    env = B2DEnv(cfg)
    env.reset()
    dev = env.device

    import omni.usd
    link_paths = list(env.robot.root_physx_view.link_paths[0])
    links = [p.rsplit("/", 1)[-1] for p in link_paths]
    assert links == list(env.robot.body_names), "link path order differs from body_names"
    meshes = visual_meshes(omni.usd.get_context().get_stage(), link_paths)

    names = [str(n) for n in data["joint_names"]]
    perm = [names.index(n) for n in env.joint_names]
    out = {"link_names": np.array(links)}
    for i, m in enumerate(meshes):
        if m is not None:
            out[f"mesh/{i}/v"], out[f"mesh/{i}/f"] = m
    origin = env.scene.env_origins
    for key in data.files:
        if not key.startswith("traj/"):
            continue
        q_all = torch.as_tensor(data[key][:, perm], dtype=torch.float32, device=dev)
        poses = []
        for s in range(0, len(q_all), env.num_envs):
            q = q_all[s:s + env.num_envs]
            n = len(q)
            if n < env.num_envs:                                   # pad the last chunk
                q = torch.cat([q, q[-1:].expand(env.num_envs - n, -1)])
            env.teleport(q=q)
            p = env.robot.data.body_pos_w - origin[:, None]
            w = env.robot.data.body_quat_w                         # wxyz
            poses.append(torch.cat([p, w[..., 1:], w[..., :1]], -1)[:n].cpu().numpy())
        out["pose/" + key[5:]] = np.concatenate(poses).astype(np.float32)
        print(f"  {args.robot} {key}: {len(q_all)} frames", flush=True)
    np.savez_compressed(args.out, **out)
    nv = sum(1 for m in meshes if m is not None)
    print(f"RESULT viewer_fk {args.robot}: {len(links)} links, {nv} with visuals -> {args.out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
