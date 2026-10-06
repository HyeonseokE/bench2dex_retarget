# experiments/

Numbers, configs and notes of the cross-hand retargeting (tracked in git). Heavy outputs (references,
SPIDER traces, rollouts, Bench2Dex HDF5 episodes, videos, logs) are in `../results/` (git-ignored).

```
experiments/
  STATUS.md                     all (task, target hand) pairs: success / failed / in progress
  ENVIRONMENT.md                machines, software versions, simulation settings (Bench2Dex physics unchanged)
  benchmark_plan.md             scope (12 tasks x 5 UR5 hands), dataset counts, pipeline, compute estimate
  stage3_settings.md            current SPIDER / env / attempt settings and why each value was chosen
  <scene>/                      the 12 UR5-demo tasks (06 12 42 07 34 60 43 76 08 44 21 27)
    README.md                   task description, source hand, target hands
    <target>/                   the 4 target hands (every UR5 hand but the source)
      README.md                 the pair, where its data is, notes
      config.yaml               stage-3 / env / attempt settings in force (generated from the code)
      STATUS.md, results.csv    per-episode progress and verdicts
```

`tools/retarget/init_experiments.py` creates the tree (keeps existing READMEs, rewrites config.yaml);
`tools/retarget/status_report.py` writes the STATUS/results files (run_queue.py does both).
Each SPIDER attempt also stores its exact settings in `results/<scene>/epNNN/<target>/spider_aK.json`.
