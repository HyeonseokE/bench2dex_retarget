"""Write the dataset card (README.md) and failures.json of every target repo of a task.

Run once the task's episodes are done (ondemand/finalize.sbatch does it after the array). The card is
built from what is actually in the repo plus the local status.json of every source episode: which
episodes are in, which never succeeded within the retry budget (left out, to be handled later).

  python tools/retarget/hf_card.py --task 06 [--targets "shadow wuji"]
"""

import argparse
import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # Bench2Dex root
from retarget import paths, robots  # noqa: E402
from retarget.task import list_episodes, load_episode  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from upload_hf import repo_id, retry, token  # noqa: E402

CARD = """---
license: other
task_categories:
- robotics
tags:
- bench2dex
- dexterous-manipulation
- bimanual
- cross-embodiment
- retargeting
- spider
pretty_name: "Bench2Dex {scene}: {target} (SPIDER retargeting)"
---

# Bench2Dex `{scene}` retargeted to UR5 + {target_long}

Teleoperated Bench2Dex demonstrations of **{scene}** (source hand: {source_long}) retargeted to
**{target_long}** with SPIDER (sampling-based physics retargeting, arXiv 2511.09484), then {made} like the
released [Bench2Dex/teleopdata](https://huggingface.co/datasets/Bench2Dex/teleopdata).

- **Episodes:** {n_up} of {n_src} source episodes ({missing_note})
- **Success:** every episode here is a stable success of Bench2Dex's own `MetricTracker` on the recorded run.
- **Generator:** [HyeonseokE/bench2dex_retarget `retarget/`](https://github.com/HyeonseokE/bench2dex_retarget/tree/main/retarget)

## Layout

```
{layout}
```

`episode_XXXXXX` is the index of the source episode in `Bench2Dex/teleopdata/dataset/{scene}`; the scene,
seeds, generalization sample (backdrop, table texture, lighting, cameras, clutter), instruction and object
bounding boxes are the source episode's, so `Bench2Dex/replay.py --restore-generalization` rebuilds the
same scene. Each file follows the Bench2Dex raw HDF5 schema (`raw_hdf5_v2`): 20 fps, Convention-A actions
(`action/commanded` = absolute joint-position targets executed after the frame; frame 0 has none),
`meta/retarget_info` records the source episode, seed and SPIDER settings.

Differences from the release: hand-specific tactile site names (as Bench2Dex defines them per robot), metric
keys of the pinned Bench2Dex revision, no `_1` re-rendered appearance variants, distractors are kinematic
(their recorded poses), and `homing_start_sim_step = -1` (the retargeted run ends with a 2 s hold, no homing).

## Not included

{missing_list}

## License

The source teleoperation data is distributed without a declared license; Bench2Dex code is MIT.
Check the Bench2Dex terms before redistribution.
"""

LAYOUT = {
    "origin": "dataset/{scene}/{target}/origin-generalization/episode_XXXXXX.hdf5   states, actions, objects, box3d, metrics (no images)",
    "replay": "dataset/{scene}/{target}/replay-generalization/episode_XXXXXX.hdf5   origin + RGB x6 (JPEG 480x640) + TacMap tactile x10 + box2d + occupancy",
}
STAGE_DIR = {"origin": "origin-generalization", "replay": "replay-generalization"}


def card_text(scene, target, source, n_up, n_src, failures, stages):
    """Dataset card for the stages in the repo ("origin" only = recorded, not rendered)."""
    missing_list = "\n".join(f"- episode_{k}: {v['state']} ({v['attempts']} attempts)" for k, v in failures.items()) or "None."
    return CARD.format(scene=scene, target=target, target_long=LONG[target], source_long=LONG[source], n_up=n_up,
                       n_src=n_src, missing_note="all included" if not failures else f"{len(failures)} left out, listed below",
                       missing_list=missing_list,
                       made="recorded and rendered" if "replay" in stages else "recorded (states only, not rendered)",
                       layout="\n".join(LAYOUT[s].format(scene=scene, target=target) for s in stages))


LONG = {"rh56dfx": "Inspire RH56DFX", "rh5dg2": "RH5DG2", "shadow": "Shadow Hand", "schunk": "Schunk SVH",
        "wuji": "Wuji Hand"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--targets", default="")
    a = ap.parse_args()
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token())
    scene = paths.task_dir(a.task).parent.name
    eps = list_episodes(a.task)
    source = robots.get(load_episode(a.task, eps[0]).robot_key).name
    targets = a.targets.split() or [r for r in robots.ROBOTS if r != source]
    for tgt in targets:
        rid = repo_id(api, scene, tgt)
        try:
            files = retry(lambda: api.list_repo_files(rid, repo_type="dataset"), "list")
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {rid}: {type(e).__name__}", flush=True)
            continue
        stages = os.environ.get("HF_STAGES", "origin replay").split()
        last = STAGE_DIR[stages[-1]]
        up = sorted({int(m.group(1)) for f in files if (m := re.search(last + r"/episode_(\d{6})\.hdf5$", f))})
        failures = {}
        for e in eps:
            if e in up:
                continue
            st = paths.RUNS / scene / f"ep{e:03d}" / tgt / "status.json"
            s = json.load(open(st)) if st.exists() else {"state": "not run"}
            failures[f"{e:06d}"] = {"state": s.get("state"), "attempts": len(s.get("attempts", []))}
        card = card_text(scene, tgt, source, len(up), len(eps), failures, stages)
        with tempfile.TemporaryDirectory() as tmp:
            open(f"{tmp}/README.md", "w").write(card)
            json.dump(failures, open(f"{tmp}/failures.json", "w"), indent=1)
            retry(lambda: api.create_commit(rid, repo_type="dataset", commit_message="dataset card",
                                            operations=[CommitOperationAdd("README.md", f"{tmp}/README.md"),
                                                        CommitOperationAdd("failures.json", f"{tmp}/failures.json")]), "card")
        print(f"RESULT card {rid}: {len(up)}/{len(eps)} episodes, {len(failures)} left out", flush=True)


if __name__ == "__main__":
    main()
