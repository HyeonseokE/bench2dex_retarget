# 06_fruit_bowl_loading: Inspire RH56DFX -> Shadow Hand

Retargeting of the 06_fruit_bowl_loading teleop demos from UR5 + Inspire RH56DFX onto UR5 + Shadow Hand.

| file | content |
|---|---|
| `config.yaml` | stage-3 settings in force (from the code defaults; each attempt also stores its own in `spider_aK.json`) |
| `STATUS.md`, `results.csv` | per-episode progress and verdicts (status_report.py) |

Data: `results/06_fruit_bowl_loading/epNNN/shadow/` (reference, kinematic, SPIDER trace/rollout, run.log), episodes: `results/dataset/shadow/06_fruit_bowl_loading/`.

## Notes

- `legacy_prototype/`: the /workspace/retarget prototype run (2026-10-05/06): 25/50 episodes, 14 success, 11 fail; data in `results/legacy_06_shadow/`, viewer https://claude.ai/artifact/RyNPfj5n9gjvZzo95GiBm8
