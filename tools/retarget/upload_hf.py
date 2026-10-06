"""Push one retargeted episode to its Hugging Face dataset repo (one commit per episode).

  repo   <namespace>/b2d-<task scene name>-<target>-retargeting   (public dataset; created on first push)
  files  dataset/<task scene name>/<target>/origin-generalization/episode_XXXXXX.hdf5
         dataset/<task scene name>/<target>/replay-generalization/episode_XXXXXX.hdf5

The layout follows Bench2Dex/teleopdata (dataset/<task>/<stage>/episode_*.hdf5) with the target hand
as an extra level. Episodes of the same task finish at different times on different array tasks, so
each pushes its own commit; files already in the repo are skipped (resubmitting is safe) and
transient Hub errors (rate limit, concurrent-commit conflict, 5xx) are retried with backoff.

  HF_TOKEN (or ~/.hf_token), HF_NAMESPACE (default: the token's user), HF_STAGES (default "origin replay")

  python tools/retarget/upload_hf.py --task 06 --episode 0 --target shadow
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # Bench2Dex root
from retarget import paths  # noqa: E402

STAGES = {"origin": "origin-generalization", "replay": "replay-generalization"}


def token():
    t = os.environ.get("HF_TOKEN")
    if not t and (Path.home() / ".hf_token").exists():
        t = (Path.home() / ".hf_token").read_text().strip()
    if not t:
        sys.exit("FATAL: no HF token (HF_TOKEN or ~/.hf_token)")
    return t


def repo_id(api, scene: str, target: str) -> str:
    ns = os.environ.get("HF_NAMESPACE") or api.whoami()["name"]
    return f"{ns}/b2d-{scene}-{target}-retargeting"


def retry(fn, what, tries=8):
    from huggingface_hub.utils import HfHubHTTPError
    for i in range(tries):
        try:
            return fn()
        except HfHubHTTPError as e:
            code = getattr(e.response, "status_code", None)
            if code not in (409, 412, 429, 500, 502, 503, 504) or i == tries - 1:
                raise
            wait = min(600, 15 * 2 ** i) * (0.5 + random.random())
            print(f"  {what}: HTTP {code}, retry {i + 1}/{tries - 1} in {wait:.0f}s", flush=True)
            time.sleep(wait)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--target", required=True)
    args = ap.parse_args()
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token())
    scene = paths.task_dir(args.task).parent.name
    rid = repo_id(api, scene, args.target)
    stages = os.environ.get("HF_STAGES", "origin replay").split()
    local = paths.RUNS / "dataset" / args.target / scene
    name = f"episode_{args.episode:06d}.hdf5"
    ops, missing = [], []
    for s in stages:
        f = local / STAGES[s] / name
        if not f.exists():
            missing.append(str(f))
            continue
        ops.append(CommitOperationAdd(path_in_repo=f"dataset/{scene}/{args.target}/{STAGES[s]}/{name}",
                                      path_or_fileobj=str(f)))
    if missing:
        sys.exit(f"FATAL: not produced yet: {missing}")
    retry(lambda: api.create_repo(rid, repo_type="dataset", private=False, exist_ok=True), "create_repo")
    have = set(retry(lambda: api.list_repo_files(rid, repo_type="dataset"), "list"))
    ops = [o for o in ops if o.path_in_repo not in have]
    if not ops:
        print(f"RESULT upload {rid}: {name} already there", flush=True)
        return
    retry(lambda: api.create_commit(rid, repo_type="dataset", operations=ops,
                                    commit_message=f"{scene} {args.target}: episode {args.episode:06d}"), "commit")
    print(f"RESULT upload {rid}: {[o.path_in_repo for o in ops]}", flush=True)


if __name__ == "__main__":
    main()
