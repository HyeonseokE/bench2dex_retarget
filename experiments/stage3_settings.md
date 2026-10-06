# Stage 3 (SPIDER) settings

Source of truth: `tools/retarget/stage3_spider.py` (CLI), `retarget/spider_env.py` (`SpiderEnvCfg`),
`tools/retarget/run_target.py` (attempt schedule). Per (task, target): `<scene>/<target>/config.yaml`;
each run records what it used in `results/<scene>/epNNN/<target>/spider_aK.json`.

## Current defaults = "v2" (2026-10-06)

The setting that retargeted 06 RH56DFX -> Shadow ep1-ep5 in a row on the prototype, and 06 ep1 and ep3
2/2 on this repo code (both dropped the first apple, re-acquired it after retries, and passed the
Bench2Dex MetricTracker check).

| setting | value | meaning |
|---|---|---|
| num_samples | 1024 | action candidates simulated per replanning step |
| horizon / knot / commit | 30 / 5 / 5 frames | plan 30 frames, knots every 5, execute 5, replan |
| iters / sigma | 5 / 0.5 | MPPI iterations and sampling std per step |
| guide_gain | 300 N/m -> 0 | virtual contact spring, annealed over the iterations; never on committed frames |
| squeeze | 0.02 m | contact targets moved into the object |
| finger_res_cap / wrist | 0.6 rad / 0.05 m, 0.5 rad | residual range around the kinematic reference |
| action_ema | 0.8 | smoothing |
| **gate_margin** | **15 frames** | residual allowed this many frames around manipulation |
| **gate_look** | **30 frames** | optimise only when manipulation is this close (= the planning horizon) |
| drop_jump | 0.025 m | an active object's error > 4 cm that grew by this much since the last commit = drop |
| **backtrack / max_retries** | **2 commits (10 frames) / 3** | after a drop go back 10 frames, iterations x(1+level), std x(1+0.5 level), up to 3 times per restart frame; then carry on. A later drop restarts from a new frame with a new budget |
| settle | 40 frames | hold the last command after the demo; success keeps being checked |
| early stop | none | the run always reaches the end of the demo |

Attempts (`run_target.py`): attempt K = seed K, num_samples 1024 * (1 + K // 2), iters 5 + K, up to 5,
stop at the first MetricTracker success.

## Rejected settings — do not retry these patterns

All on 06 RH56DFX -> Shadow (2026-10-05/06). The common lesson: **an object that is dropped is usually
re-acquired later** (prototype successes dropped one object 6-18 times, e.g. ep002 banana x18,
ep014 apple x8), so anything that limits or ends the search after a few drops, or shortens the
pre-grasp optimisation window, costs successes.

| pattern | what it was | result | why it failed |
|---|---|---|---|
| Narrow sampling window | gate_margin 10, gate_look 10 (also 13 / 26) | 9/14 = 64% vs 5/6 = 83% with 15 / 30 | less time to shape the fingers before contact; most failures on the first apple |
| Abort after N retries | stop the episode when a segment failed 5 (or 7) retries | 06 ep20-24: 5/5 failed, every one stopped exactly at the cap | kills runs that would have re-acquired the object |
| Per-segment retry budget | 5 (then 7, then 12) retries for a whole manipulation segment | used up quickly, then abort | the v2 rule gives each new restart frame its own 3 retries |
| "Lagging" drop trigger | also count error > 6 cm and growing as a drop | budget consumed by normal lifting transients | error grows briefly during a lift that succeeds |
| Fine backtrack schedule | 10, 10, 12, 14, 16, 18, 20 frames back (2-frame steps, mid-commit restarts) | ep024: bowl lost 7x then aborted; never shown better | restarts mid-commit with a plan misaligned by 1-4 frames; no benefit observed |
| finger_res_cap 0.25 / 0.4, no squeeze | earlier SPIDER configs | apples not lifted (fingertips 1.6-2.7 cm off the surface) | the kinematic Shadow grasp is loose; needs 0.6 rad and contact targets 2 cm inside |

Also not a setting but a trap: running more than 3 Isaac processes on a 24 GB GPU (CUDA OOM corrupted
PhysX state and produced a fake success on 06 ep4; stage 3 now stops with an error instead).
