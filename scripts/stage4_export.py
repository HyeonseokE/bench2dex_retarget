"""Stage 4: a SPIDER rollout as a Bench2Dex episode HDF5 of the target robot.

The source episode's meta (scene, generalization sample, timing) is kept; robot_key, robot qpos and
task-object states are replaced by the rollout, so Bench2Dex's replay.py renders it like any other
episode:

  python replay.py --hdf5 <out.hdf5> --restore-generalization --enable-rgb --enable-tactile --headless

Distractors were not simulated; their recorded poses are copied through. Pure numpy/h5py.

  python scripts/stage4_export.py --task 06 --episode 0 --target shadow
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from b2dr import paths, robots  # noqa: E402
from b2dr.task import wxyz_to_xyzw  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--only_success", action="store_true", help="export nothing for a failed rollout")
    args = ap.parse_args()

    d = paths.run_dir(args.task, args.episode) / args.target
    ro = np.load(d / "spider_rollout.npz", allow_pickle=True)   # stage 3's default --tag
    res = json.loads(str(ro["result"]))
    if args.only_success and not res["task_success"]:
        print(f"skip export: {d} did not succeed")
        return
    spec = robots.get(args.target)
    src_path = paths.episode_path(args.task, args.episode)
    q = ro["q"].astype(np.float32)
    names = [str(n) for n in ro["joint_names"]]
    obj = ro["obj"]
    obj_ids = [str(o) for o in ro["obj_ids"]]
    T = len(q)
    out = d / f"episode_{args.episode:06d}_{spec.name}.hdf5"
    with h5py.File(src_path, "r") as src, h5py.File(out, "w") as dst:
        n_src = len(src["time/sim_step"])
        s = 1                                          # rollout frame k == source frame k + 1
        fi = np.minimum(np.arange(T) + s, n_src - 1)
        m = dst.create_group("meta")
        for k in src["meta"]:
            src.copy(src[f"meta/{k}"], m, name=k)
        for k, v in {"robot_key": spec.key, "frame_count": T, "has_rgb": False, "has_tactile": False,
                     "homing_start_sim_step": -1, "source_domain": "sim_retarget_spider",
                     "success": bool(res["task_success"])}.items():
            if k in m:
                del m[k]
            m[k] = v
        m["retarget_info"] = json.dumps({"source_episode": str(src_path), "result": res})
        stride = int(src["meta/step_stride"][()])
        dst["time/frame_index"] = np.arange(T, dtype=np.int32)
        dst["time/sim_step"] = src["time/sim_step"][s] + stride * np.arange(T, dtype=np.int64)
        dst["robot/joint_names"] = np.array(names, dtype=object).astype("S")
        dst["robot/qpos"] = q
        dst["robot/qvel"] = (np.gradient(q, axis=0) * int(src["meta/fps"][()])).astype(np.float32)
        dst["action/commanded"] = ro["q_target"].astype(np.float32)
        dst["action/action_names"] = np.array(names, dtype=object).astype("S")
        for k in src["objects"]:
            g = dst.create_group(f"objects/{k}")
            if k in obj_ids:
                g["pose_world"] = wxyz_to_xyzw(obj[:, obj_ids.index(k)]).astype(np.float32)
                if f"art/{k}" in ro:
                    g["qpos"] = ro[f"art/{k}"].astype(np.float32)
                    g["joint_names"] = src[f"objects/{k}/joint_names"][:]
            else:
                g["pose_world"] = src[f"objects/{k}/pose_world"][fi]
    print(f"wrote {out}: {T} frames, robot={spec.key}, success={res['task_success']}")


if __name__ == "__main__":
    main()
