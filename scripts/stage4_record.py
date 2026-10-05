"""Stage 4: re-execute the SPIDER plan open-loop and record it as a Bench2Dex episode.

SPIDER's committed rollout went through a snapshot restore at every commit. A dataset should be what
a teleop recording is: one continuous execution in which joint targets go into physics. So a fresh
single-env scene is reset to the plan's initial state and the stored joint targets are executed
in order (no residual sampling, no guidance), while b2dr.recorder records it the way Bench2Dex's
DataCollector does. The verdict is MetricTracker's stable success on THIS execution.

GPU PhysX is not bit-reproducible across scene sizes, so the re-execution can drift from the
optimised rollout; the drift is reported, and only a re-execution that succeeds is written to the
dataset tree (failures go to dataset_failed/ when --keep_failed).

  out: $B2DR_RUNS/dataset/<robot>/<scene>/origin-generalization/episode_<ep>.hdf5

  python scripts/stage4_record.py --task 06 --episode 0 --target shadow --headless
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
ap.add_argument("--tag", default="spider", help="which stage-3 plan: <tag>_rollout.npz")
ap.add_argument("--keep_failed", action="store_true")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from b2dr import paths  # noqa: E402
from b2dr.recorder import Bench2DexRecorder  # noqa: E402
from b2dr.sim_env import B2DEnv, B2DEnvCfg  # noqa: E402


def main():
    out_dir = paths.run_dir(args.task, args.episode) / args.target
    res_path = out_dir / f"{args.tag}_record.json"
    if res_path.exists():
        print(f"skip: {res_path} exists", flush=True)
        return
    plan = np.load(out_dir / f"{args.tag}_rollout.npz", allow_pickle=True)
    cfg = B2DEnvCfg(task=args.task, episode=args.episode, robot=args.target)
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    env = B2DEnv(cfg)
    env.reset()
    dev = env.device
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)[None]
    names = [str(n) for n in plan["joint_names"]]
    assert names == env.joint_names, "plan was made for a different joint order"

    # the plan's initial state (zero velocities, as SPIDER's reset)
    q0 = t(plan["init/q"])
    env.robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
    env.q_target = t(plan["init/q_target"])
    env.robot.set_joint_position_target(env.q_target)
    for j, (k, a) in enumerate(env.objects.items()):
        p = t(plan[f"init/obj/{k}/pose"])
        p[:, :3] += env.scene.env_origins
        a.write_root_pose_to_sim(p)
        a.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
        if f"init/obj/{k}/jp" in plan:
            jp = t(plan[f"init/obj/{k}/jp"])
            a.write_joint_state_to_sim(jp, torch.zeros_like(jp))
            a.set_joint_position_target(t(plan[f"init/obj/{k}/jt"]))
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=0.0)

    rec = Bench2DexRecorder(env, paths.episode_path(args.task, args.episode))
    rec.observe()                                                 # frame 0, no action
    drift = []
    ref_obj = plan["obj"]
    for i, cmd in enumerate(plan["cmd"]):
        rec.set_action(cmd)
        env.q_target = t(cmd)
        for _ in range(env.cfg.decimation):
            env._apply_action()                                   # targets + gravity compensation
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            rec.after_physics_step()
        rec.observe()
        roots, _ = env.object_roots()
        drift.append(np.linalg.norm(roots[0, :, :3].cpu().numpy() - ref_obj[i + 1][:, :3], axis=-1))
    drift = np.array(drift)

    res3 = json.loads(str(plan["result"]))
    info = {"source_episode": str(paths.episode_path(args.task, args.episode)), "source_robot": res3["source"],
            "method": "spider", "seed": res3.get("seed"), "spider_args": res3.get("args"), "env_cfg": res3.get("env_cfg"),
            "spider_scene_success": res3.get("scene_success")}
    fin = rec.finalize()
    success = bool(fin[0].stable_success) if fin else False
    h5 = None
    if success or args.keep_failed:
        root = paths.RUNS / ("dataset" if success else "dataset_failed")
        if args.tag != "spider":
            root = out_dir / f"{args.tag}_{root.name}"
        h5, success = rec.write(root, args.episode, {"retarget_info": json.dumps(info)})
    res = {"task": res3["task"], "episode": args.episode, "source": res3["source"], "target": args.target,
           "success": success, "frames": len(rec.frames), "episode_hdf5": str(h5) if h5 else None,
           "drift_vs_plan_cm": {"max": round(100 * float(drift.max()), 2), "final_mean": round(100 * float(drift[-1].mean()), 2)},
           "spider_scene_success": res3.get("scene_success")}
    json.dump(res, open(res_path, "w"), indent=1)
    print(f"RESULT stage4 {res['task']} ep{args.episode} {res['source']}->{args.target}: "
          f"{'SUCCESS' if success else 'fail'} drift max {res['drift_vs_plan_cm']['max']} cm -> {h5}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
