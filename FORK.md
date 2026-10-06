# Fork notes (HyeonseokE/Bench2Dex)

Rule: **upstream files are never edited.** Everything this fork adds lives in new files, so
`git merge upstream/main` cannot conflict. Behaviour changes go in wrappers (or runtime patches
inside our own modules), never in the authors' code.

| path | what |
|---|---|
| `retarget/` | SPIDER cross-embodiment retargeting library ([README](retarget/README.md)) |
| `tools/retarget/` | retargeting entry points (stage1–5, run_target, HF upload, audit) |
| `ondemand/` | SLURM jobs for the pro6000 cluster ([README](ondemand/README.md)) |
| `policy/MyPolicy/` | own policy, in the authors' plugin location (`policy.<name>.deploy_policy`) |
| `tools/eval/` | eval helpers: `act_shared_gpu.sh` (wraps `policy/ACT/eval_double_env.sh`), `summarize_eval.py` |
| `FORK.md` | this file |

Check that no upstream file is modified (must print nothing):

    git fetch upstream && git diff --name-status upstream/main HEAD | grep -v '^A'

Update from the authors:

    git fetch upstream && git merge upstream/main && git push origin main

Then run `ondemand/env_check.sbatch` once: a clean merge can still change an API that
`retarget/robots.py` (spawner), `retarget/sim_env.py` (scene) or `retarget/recorder.py` (recording) uses.
