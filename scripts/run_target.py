"""Everything for one (task, episode, target hand): retarget until it succeeds, record, render, upload.

  stage 2   kinematic retarget (once)
  stage 3+4 SPIDER, then the Bench2Dex record of its committed execution; the verdict is
            MetricTracker's stable success. On failure retry with a new seed and more search:
              attempt a: --seed a, --num_samples 1024 * (1 + a // 2), --iters 5 + a
            up to --max_attempts. An episode that never succeeds is reported, not uploaded.
  stage 5   render the successful episode like the released replay data (RGB x6, TacMap, labels)
  upload    one commit to <ns>/b2d-<scene>-<target>-retargeting (scripts/upload_hf.py)

Every step skips what is already on disk, so a resubmitted job resumes where it stopped. Each Isaac
stage runs in its own process (an Isaac Lab scene cannot be rebuilt in one process); the interpreter
is $PY (the cluster's /isaac-sim/python.sh) or this one.

  status: $B2DR_RUNS/<scene>/epNNN/<target>/status.json

  python scripts/run_target.py --task 06 --episode 0 --target shadow [--max_attempts 5] [--no_upload]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from b2dr import paths  # noqa: E402

PY = os.environ.get("PY", sys.executable)


def run(log: Path, cmd: list, timeout: int) -> int:
    with open(log, "a") as f:
        f.write(f"\n--- {time.strftime('%F %T')} {' '.join(map(str, cmd))}\n")
        f.flush()
        try:
            return subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            f.write(f"--- TIMEOUT after {timeout}s\n")
            return 124


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--max_attempts", type=int, default=5)
    ap.add_argument("--spider_args", default="", help="extra stage-3 arguments for every attempt")
    ap.add_argument("--no_render", action="store_true")
    ap.add_argument("--no_upload", action="store_true")
    a = ap.parse_args()

    d = paths.run_dir(a.task, a.episode) / a.target
    d.mkdir(parents=True, exist_ok=True)
    log, status_path = d / "run.log", d / "status.json"
    status = json.load(open(status_path)) if status_path.exists() else {"attempts": []}
    scene = paths.task_dir(a.task).parent.name
    status.update({"task": scene, "episode": a.episode, "target": a.target})

    def save(**kw):
        status.update(kw)
        json.dump(status, open(status_path, "w"), indent=1)

    common = ["--task", a.task, "--episode", str(a.episode), "--target", a.target]
    try:
        body(a, d, log, status, save, common)
    finally:
        print(f"RESULT run_target {scene} ep{a.episode} -> {a.target}: {status.get('state')}", flush=True)


def body(a, d, log, status, save, common):
    if run(log, [PY, HERE / "stage2_kinematic.py", *common, "--headless"], 3600) != 0:
        return save(state="stage2_failed")

    good = next((x for x in status["attempts"] if x.get("success")), None)
    for att in range(a.max_attempts):
        if good:
            break
        tag = f"spider_a{att}"
        rec_path = d / f"{tag}_record.json"
        if not rec_path.exists():
            extra = ["--seed", str(att), "--num_samples", str(1024 * (1 + att // 2)), "--iters", str(5 + att)]
            rc = run(log, [PY, HERE / "stage3_spider.py", *common, "--tag", tag, *extra, *a.spider_args.split(),
                           "--headless"], 6 * 3600)
            if rc != 0 or not (d / f"{tag}_trace.pkl.gz").exists():
                status["attempts"].append({"tag": tag, "success": False, "error": f"stage3 rc={rc}"})
                save(state="retrying")
                continue
            if run(log, [PY, HERE / "stage4_record.py", *common, "--tag", tag], 1800) != 0:
                status["attempts"].append({"tag": tag, "success": False, "error": "stage4 failed"})
                save(state="retrying")
                continue
        rec = json.load(open(rec_path))
        entry = {"tag": tag, "success": rec["success"], "episode_hdf5": rec["episode_hdf5"]}
        status["attempts"] = [x for x in status["attempts"] if x["tag"] != tag] + [entry]
        save(state="retrying" if not rec["success"] else "recorded")
        if rec["success"]:
            good = entry
    if not good:
        return save(state="failed_all_attempts")

    origin = Path(good["episode_hdf5"])
    replay = origin.parent.parent / "replay-generalization" / origin.name
    if not a.no_render and not replay.exists():
        env = {**os.environ, "PY": PY}
        with open(log, "a") as f:
            rc = subprocess.run(["bash", str(HERE / "stage5_render.sh"), str(origin.parent), origin.name],
                                stdout=f, stderr=subprocess.STDOUT, env=env, timeout=4 * 3600).returncode
        if rc != 0 or not replay.exists():
            return save(state="render_failed")
    save(state="rendered" if replay.exists() else "recorded")
    if a.no_upload or a.no_render:
        return
    if run(log, [PY, HERE / "upload_hf.py", *common], 3 * 3600) != 0:
        return save(state="upload_failed")
    save(state="uploaded")


if __name__ == "__main__":
    main()
