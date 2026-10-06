"""Retarget whole tasks on a plain GPU server (no SLURM): a queue of (task, episode, target) jobs
spread over the local GPUs, the dev-box counterpart of ondemand/main_job.sbatch.

Per episode: stage 1 once, then tools/retarget/run_target.py per target (stage 2, SPIDER with
retries, Bench2Dex record, render, optional HF upload). Every step skips finished work, so the
command can be stopped and re-run. Progress: $B2DR_RUNS/STATUS.md (tools/retarget/status_report.py).

  source tools/retarget/local/env.sh
  python tools/retarget/run_queue.py --tasks 06 12 42 --gpus 0 1 --per_gpu 2 --no_upload

Memory: one SPIDER process with 1024 samples takes ~5-6 GB; 3 per 24 GB GPU is the safe maximum
(a CUDA OOM corrupts PhysX state; stage 3 then stops with an error).
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from retarget import paths  # noqa: E402

ALL_ROBOTS = ["rh56dfx", "rh5dg2", "shadow", "schunk", "wuji"]
UR5_TASKS = ["06", "12", "42", "07", "34", "60", "43", "76", "08", "44", "21", "27"]
FINAL = {"recorded", "rendered", "uploaded", "failed_all_attempts", "stage2_failed"}


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=UR5_TASKS)
    ap.add_argument("--episodes", type=int, nargs="+", default=list(range(50)), help="episode indices")
    ap.add_argument("--targets", nargs="*", default=None, help="default: every UR5 hand but the source")
    ap.add_argument("--gpus", nargs="+", default=["0"])
    ap.add_argument("--per_gpu", type=int, default=2, help="concurrent targets per GPU")
    ap.add_argument("--max_attempts", type=int, default=5)
    ap.add_argument("--spider_args", default="")
    ap.add_argument("--no_render", action="store_true")
    ap.add_argument("--no_upload", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    slots = [g for g in a.gpus for _ in range(a.per_gpu)]
    free = list(slots)
    lock = threading.Condition()
    py = os.environ.get("PY", sys.executable)
    logdir = paths.RUNS / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    threads = []
    reporter = subprocess.Popen([py, str(HERE / "status_report.py"), "--loop", "120"]) if not a.dry_run else None

    def acquire():
        with lock:
            while not free:
                lock.wait()
            return free.pop(0)

    def release(g):
        with lock:
            free.append(g)
            lock.notify()

    def run(cmd, gpu, logf, timeout):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PY": py}
        with open(logf, "a") as f:
            try:
                return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=ROOT, timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                return 124

    def target_job(task, ep, tgt, gpu):
        try:
            cmd = [py, str(HERE / "run_target.py"), "--task", task, "--episode", str(ep), "--target", tgt,
                   "--max_attempts", str(a.max_attempts), "--spider_args", a.spider_args]
            cmd += ["--no_render"] * a.no_render + ["--no_upload"] * a.no_upload
            run(cmd, gpu, logdir / f"{task}_ep{ep:03d}_{tgt}.log", 48 * 3600)
            st = paths.run_dir(task, ep) / tgt / "status.json"
            log(f"{task} ep{ep} -> {tgt}: {json.load(open(st)).get('state') if st.exists() else 'no status'}")
        finally:
            release(gpu)

    for task in a.tasks:
        for ep in a.episodes:
            try:
                epdir = paths.run_dir(task, ep)
            except Exception as e:  # noqa: BLE001  (episode not downloaded)
                log(f"skip {task} ep{ep}: {e}")
                continue
            ref = epdir / "reference.json"
            if not ref.exists():
                if a.dry_run:
                    log(f"would run stage 1 for {task} ep{ep}")
                    continue
                gpu = acquire()
                rc = run([py, str(HERE / "stage1_reference.py"), "--task", task, "--episode", str(ep), "--headless"],
                         gpu, logdir / f"{task}_ep{ep:03d}_stage1.log", 3600)
                release(gpu)
                if rc != 0 or not ref.exists():
                    log(f"stage 1 failed for {task} ep{ep} (see logs/)")
                    continue
            src = json.load(open(ref)).get("source")
            for tgt in (a.targets or [r for r in ALL_ROBOTS if r != src]):
                st = epdir / tgt / "status.json"
                if st.exists() and json.load(open(st)).get("state") in FINAL:
                    continue
                if a.dry_run:
                    log(f"would run {task} ep{ep} {src} -> {tgt}")
                    continue
                gpu = acquire()
                log(f"start {task} ep{ep} {src} -> {tgt} on GPU {gpu}")
                th = threading.Thread(target=target_job, args=(task, ep, tgt, gpu), daemon=False)
                th.start()
                threads.append(th)
                time.sleep(20)                       # stagger Kit start-ups
    for th in threads:
        th.join()
    if reporter:
        reporter.terminate()
        subprocess.run([py, str(HERE / "status_report.py")])
    log("ALL DONE")


if __name__ == "__main__":
    main()
