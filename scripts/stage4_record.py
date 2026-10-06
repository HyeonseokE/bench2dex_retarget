"""Stage 4: write the committed SPIDER execution as a Bench2Dex episode (state record, no re-simulation).

Reads the state trace stage 3 kept of env 0 (every physics step) and writes it with Bench2Dex's own
collector pieces (b2dr.recorder.write_episode): Convention-A actions, box3d labels, MetricTracker on
every physics-step state, HDF5EpisodeWriter. The verdict is MetricTracker's stable success, the
benchmark's official one. Pure Python (no Isaac): Bench2Dex's metric and writer code only.

  out: $B2DR_RUNS/dataset/<robot>/<scene>/origin-generalization/episode_<ep>.hdf5   (success)
       $B2DR_RUNS/dataset_failed/...                                                  (--keep_failed)

  python scripts/stage4_record.py --task 06 --episode 0 --target shadow
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from b2dr import paths, recorder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--tag", default="spider", help="which stage-3 run: <tag>_trace.pkl.gz")
    ap.add_argument("--keep_failed", action="store_true")
    ap.add_argument("--headless", action="store_true", help="ignored (uniform stage command line)")
    args = ap.parse_args()

    out_dir = paths.run_dir(args.task, args.episode) / args.target
    res_path = out_dir / f"{args.tag}_record.json"
    if res_path.exists():
        print(f"skip: {res_path} exists", flush=True)
        return
    trace = recorder.load_trace(out_dir / f"{args.tag}_trace.pkl.gz")
    res3 = json.load(open(out_dir / f"{args.tag}.json"))
    info = {"source_episode": str(paths.episode_path(args.task, args.episode)), "source_robot": res3["source"],
            "method": "spider", "seed": res3.get("seed"), "spider_args": res3.get("args"),
            "env_cfg": res3.get("env_cfg"), "record": "state trace of the committed execution"}
    base = paths.RUNS if args.tag == "spider" else out_dir / args.tag
    h5, success, payload = recorder.write_episode(trace, args.task, args.episode, base / "dataset",
                                                  {"retarget_info": json.dumps(info)})
    if h5 is None and args.keep_failed:
        h5, success, payload = recorder.write_episode(trace, args.task, args.episode, base / "dataset_failed",
                                                      {"retarget_info": json.dumps(info)}, only_success=False)
    res = {"task": res3["task"], "episode": args.episode, "source": res3["source"], "target": args.target,
           "success": success, "frames": len(trace["frames"]), "episode_hdf5": str(h5) if h5 else None,
           "first_stable_success_step": (payload or {}).get("first_stable_success_step"),
           "spider_scene_success": res3.get("scene_success")}
    json.dump(res, open(res_path, "w"), indent=1)
    print(f"RESULT stage4 {res['task']} ep{args.episode} {res['source']}->{args.target}: "
          f"{'SUCCESS' if success else 'fail'} -> {h5}", flush=True)


if __name__ == "__main__":
    main()
