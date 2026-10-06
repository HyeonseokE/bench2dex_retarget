# Stage 3 (SPIDER) settings — current defaults and why

Source of truth: `tools/retarget/stage3_spider.py` (CLI), `retarget/spider_env.py` (`SpiderEnvCfg`),
`tools/retarget/run_target.py` (attempt schedule). The values per (task, target) are in
`<scene>/<target>/config.yaml`; each run records what it used in `results/<scene>/epNNN/<target>/spider_aK.json`.

## Optimiser (stage3_spider.py)

| setting | value | note |
|---|---|---|
| num_samples | 1024 | parallel envs sampled per replanning step |
| horizon / knot / commit | 30 / 5 / 5 frames | plan 30 frames, knots every 5, execute 5 |
| iters | 5 | MPPI iterations per step (x(1+level) after a drop, level <= 4) |
| sigma | 0.5 | sampling std, x(1 + 0.5 level) after a drop |
| guide_gain | 300 N/m | virtual contact spring, annealed to 0; committed frames run without it |
| squeeze | 0.02 m | contact targets moved into the object (loose Shadow grasps on 06) |
| drop_jump | 0.025 m | error jump per commit that counts as a drop |
| back_frames / back_max | 10 / 20 | backtrack from the segment's first drop: 10, 10, 12, 14, 16, 18, 20 frames (user, 10-06) |
| abort_on_fail | 0 | keep going after the last retry: prototype successes often recovered after 6-18 drops; aborting failed 06 ep20-24 |
| gate_look | 26 frames | optimise only when manipulation is this close (user: 13-frame window, 10-06) |
| settle | 40 frames | hold the last command after the demo, success keeps being checked |

## Environment (SpiderEnvCfg)

| setting | value | note |
|---|---|---|
| gate_margin / gate_ramp | 13 / 5 frames | residual allowed around manipulation only (15 -> 10 lowered 06 success 83% -> 64%; user chose 13) |
| finger_res_cap | 0.6 rad | finger residual around the reference (0.4 could not hold apples) |
| wrist_trans_scale / wrist_rot_scale | 0.05 m / 0.5 rad | wrist residual |
| action_ema | 0.8 | smoothing (arm jitter on the RL variant) |

## Attempts (run_target.py)

Attempt K (K = 0..max_attempts-1, default 5): `--seed K`, `--num_samples 1024 * (1 + K // 2)`, `--iters 5 + K`,
until MetricTracker reports stable success.

Physics: Bench2Dex robot and scene settings, unchanged (see `ENVIRONMENT.md`).
