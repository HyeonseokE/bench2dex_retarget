"""Physics replay of the stage-2 kinematic retarget: the target robot executes the kinematic joint path
(PD targets = kinematic q every control step, no residual, no guidance, objects free), so the objects
move only through the target hand's contacts. The baseline SPIDER improves on.

  python tools/retarget/kin_replay.py --task 06 --episode 0 --target wuji --headless

Output: results/<scene>/epNNN/<target>/kinreplay_rollout.npz and kinreplay.json, with the same keys as
stage 3's <tag>_rollout.npz / <tag>.json (q, cmd, obj, err, success, joint_names, obj_ids, result).
"""

import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--episode", type=int, required=True)
ap.add_argument("--target", required=True)
ap.add_argument("--settle", type=int, default=40, help="frames holding the last pose after the demo")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from retarget import paths  # noqa: E402
from retarget.spider_env import SpiderEnv, SuccessTracker, make_cfg  # noqa: E402


def obj_err(env):
    f = env.f()
    roots, _ = env.object_roots()
    e = (env.ref_root[f][0, :, :3] - roots[0, :, :3]).norm(dim=-1)
    for k, v in env.object_qpos().items():
        j = env.obj_ids.index(k)
        e[j] = e[j] + (env.ref_art[k][f][0] - v[0]).abs().sum()
    return e


def main():
    out_dir = paths.run_dir(args.task, args.episode) / args.target
    env = SpiderEnv(make_cfg(args.task, args.episode, args.target, 1, args.device))
    T = env.T
    env.reset()
    env.hold, env.guide_gain = False, 0.0
    tracker = SuccessTracker(env.scene_def, env.step_dt)
    rec = {"q": [], "cmd": [], "obj": [], "err": [], "art": []}

    def record():
        roots, _ = env.object_roots()
        rec["q"].append(env.robot.data.joint_pos[0].cpu().numpy())
        rec["cmd"].append(env.q_target[0].cpu().numpy())
        rec["obj"].append(roots[0].cpu().numpy())
        rec["art"].append({k: v[0].cpu().numpy() for k, v in env.object_qpos().items()})
        rec["err"].append(obj_err(env).cpu().numpy())
        tracker.update(env.success_states(0))

    t0 = time.time()
    record()
    for f in range(T - 1 + args.settle):
        env.q_target[:] = env.ref_q[min(f + 1, T - 1)]
        env.commit_step(None)                       # holds q_target for one control step (3 physics steps)
        if f + 1 < T:
            env.frame += 1
        record()
        if (f + 1) % 100 == 0:
            print(f"  frame {f + 1}/{T} obj err cm {np.round(100 * rec['err'][-1], 1).tolist()}", flush=True)

    er = np.array(rec["err"])
    stable = tracker.stable_frame()
    res = {"task": env.meta["task"], "episode": args.episode, "source": env.meta["source"], "target": args.target,
           "policy": "kinematic_replay", "frames": len(er), "settle_frames": args.settle,
           "scene_success": stable is not None, "scene_stable_success_frame": stable,
           "instant_success_frames": int(sum(tracker.history)), "minutes": round((time.time() - t0) / 60, 1),
           "obj_err_mean_cm": dict(zip(env.obj_ids, (100 * er.mean(0)).round(2).tolist())),
           "obj_err_final_cm": dict(zip(env.obj_ids, (100 * er[-1]).round(2).tolist())), "args": vars(args)}
    res["task_success"] = res["scene_success"]
    art = {f"art/{k}": np.array([a[k] for a in rec["art"]]) for k in env.art_ids}
    np.savez_compressed(out_dir / "kinreplay_rollout.npz", q=np.array(rec["q"]), cmd=np.array(rec["cmd"]),
                        obj=np.array(rec["obj"]), err=er, success=np.array(tracker.history),
                        joint_names=np.array(env.joint_names), obj_ids=np.array(env.obj_ids),
                        result=np.array(json.dumps(res)), **art)
    json.dump(res, open(out_dir / "kinreplay.json", "w"), indent=1)
    print(f"RESULT kin_replay {res['task']} ep{args.episode} -> {args.target}: "
          f"{'SUCCESS' if res['task_success'] else 'fail'}, final obj err cm {res['obj_err_final_cm']}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
