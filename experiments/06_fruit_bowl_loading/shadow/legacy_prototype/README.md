# Legacy prototype run: task 06 RH56DFX -> Shadow

`/workspace/retarget` prototype (`spider_retarget.py`), 2026-10-05/06, stopped when the work moved to
this repo. 25 of 50 episodes finished: 14 success, 11 fail (ep020-024 failed because of the then
abort-after-retries rule). Settings changed during the run (sampling window 15 -> 10 -> 13 frames,
retry rules); each `episodes/epNNN.json` holds the arguments it ran with.

- `STATUS.md`, `results.csv`, `episodes/*.json`: verdicts, failed conditions, object errors, timing.
- Data (rollouts, logs, web viewer source, references): `results/legacy_06_shadow/`.
- Web viewer: https://claude.ai/artifact/RyNPfj5n9gjvZzo95GiBm8
