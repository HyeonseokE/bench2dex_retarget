"""Push a task's retargeted episodes for one target hand -- only if EVERY source episode succeeded
(or, with --allow_partial, whatever succeeded, listing the rest in failures.json).

Per target: the local recorded episodes $B2DR_RUNS/dataset/<target>/<scene>/<stage>/episode_XXXXXX.hdf5 for
source episodes 0..N-1 (N = --episodes) must all exist, each with a status.json whose successful attempt
is the file. Then one commit to <ns>/b2d-<scene>-<target>-retargeting adds every episode, the dataset card
(README.md) and an empty failures.json. If any episode is missing nothing is pushed and the missing ones are
listed (exit code 2). Episodes already in the Hub repo count as present, so a later run on another machine
that retargeted only the missing ones completes the repo (and rewrites the card as 50/50).

  HF_STAGES (default "origin": recorded HDF5 only, no rendering) selects what is pushed.

  python tools/retarget/upload_task.py --task 07 [--targets "shadow wuji"] [--episodes 50] [--dry_run]
"""

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # Bench2Dex root
from retarget import paths, robots  # noqa: E402
from retarget.task import list_episodes, load_episode  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_card import STAGE_DIR, card_text  # noqa: E402
from upload_hf import repo_id, retry, token  # noqa: E402


def check(scene: str, target: str, episodes: list[int], stages: list[str]) -> tuple[list, dict]:
    """(files to push as (local, path_in_repo), {episode: why it is missing})."""
    files, missing = [], {}
    for e in episodes:
        name = f"episode_{e:06d}.hdf5"
        st = paths.RUNS / scene / f"ep{e:03d}" / target / "status.json"
        s = json.load(open(st)) if st.exists() else {"state": "not run"}
        good = next((x for x in s.get("attempts", []) if x.get("success")), None)
        if not good:
            missing[f"{e:06d}"] = {"state": s.get("state"), "attempts": len(s.get("attempts", []))}
            continue
        for stg in stages:
            f = paths.RUNS / "dataset" / target / scene / STAGE_DIR[stg] / name
            if not f.exists():
                missing[f"{e:06d}"] = {"state": f"{stg} file missing", "attempts": len(s.get("attempts", []))}
                break
            files.append((str(f), f"dataset/{scene}/{target}/{STAGE_DIR[stg]}/{name}"))
    return files, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--targets", default="")
    ap.add_argument("--episodes", type=int, default=50, help="source episodes 0..N-1 that must all succeed")
    ap.add_argument("--allow_partial", action="store_true", help="push what succeeded even if some episodes are missing")
    ap.add_argument("--dry_run", action="store_true", help="check only, push nothing")
    a = ap.parse_args()

    scene = paths.task_dir(a.task).parent.name
    have_src = set(list_episodes(a.task))
    episodes = list(range(a.episodes))
    if not set(episodes) <= have_src:
        sys.exit(f"FATAL: source episodes missing locally: {sorted(set(episodes) - have_src)}")
    source = robots.get(load_episode(a.task, episodes[0]).robot_key).name
    targets = a.targets.split() or [r for r in robots.ROBOTS if r != source]
    stages = os.environ.get("HF_STAGES", "origin").split()

    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi(token=token()) if (not a.dry_run or os.environ.get("HF_TOKEN")) else None
    rc = 0
    for tgt in targets:
        files, missing = check(scene, tgt, episodes, stages)
        have = set()
        if api is not None:
            rid = repo_id(api, scene, tgt)
            try:
                have = set(retry(lambda: api.list_repo_files(rid, repo_type="dataset"), "list"))
            except Exception:  # noqa: BLE001  (repo not created yet)
                have = set()
            on_hub = [k for k in missing
                      if all(f"dataset/{scene}/{tgt}/{STAGE_DIR[s]}/episode_{k}.hdf5" in have for s in stages)]
            for k in on_hub:
                missing.pop(k)
        n_ok = a.episodes - len(missing)
        if missing and not a.allow_partial:
            rc = 2
            print(f"RESULT upload_task {scene} -> {tgt}: NOT UPLOADED, {n_ok}/{a.episodes} succeeded; missing "
                  + ", ".join(f"{k} ({v['state']})" for k, v in missing.items()), flush=True)
            continue
        if a.dry_run:
            print(f"RESULT upload_task {scene} -> {tgt}: ready, {n_ok}/{a.episodes} ({len(files)} local files)"
                  + (f", missing {sorted(missing)}" if missing else "") + " [dry run]", flush=True)
            continue
        rid = repo_id(api, scene, tgt)
        retry(lambda: api.create_repo(rid, repo_type="dataset", private=False, exist_ok=True), "create_repo")
        with tempfile.TemporaryDirectory() as tmp:
            open(f"{tmp}/README.md", "w").write(card_text(scene, tgt, source, n_ok, a.episodes, missing, stages))
            json.dump(missing, open(f"{tmp}/failures.json", "w"), indent=1)
            ops = [CommitOperationAdd(p, f) for f, p in files if p not in have]
            ops += [CommitOperationAdd("README.md", f"{tmp}/README.md"), CommitOperationAdd("failures.json", f"{tmp}/failures.json")]
            retry(lambda: api.create_commit(rid, repo_type="dataset", operations=ops,
                                            commit_message=f"{scene} {tgt}: {n_ok}/{a.episodes} episodes ({' + '.join(stages)})"),
                  "commit")
        print(f"RESULT upload_task {scene} -> {tgt}: UPLOADED {n_ok}/{a.episodes} to {rid} ({len(ops) - 2} new files)"
              + (f"; missing {sorted(missing)}" if missing else ""), flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
