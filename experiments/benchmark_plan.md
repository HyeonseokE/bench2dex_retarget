# Cross-hand benchmark dataset plan (Bench2Dex, UR5)

Target: a cross-hand (embodiment, not arm) benchmark, made by retargeting Bench2Dex teleop demos onto
the other hands that Bench2Dex mounts on the **same UR5 arm**, so that only the hand differs.

## 1. Bench2Dex at a glance (checked 2026-10-06)

| Item | Paper | Data / code |
|---|---|---|
| Tasks | 26 | 26 task datasets on HF `Bench2Dex/teleopdata`; 27 scene yamls (`86_short_jigsaw_puzzle` has no demos) |
| Robot embodiments | 12 | 12 registered in `robots/__init__.py` |
| Demos | ~1.3K (26 x 50) | per task `origin-generalization/`: `episode_000000..049.hdf5` + the same 50 indices with a `_1` suffix. The `_1` files are re-renders of the same trajectories (checked on task 06; consistent with 26 x 50 = 1.3K), so a task has **50 unique episodes** |

- Each Bench2Dex robot is a fixed **arm + hand** asset. Scene yamls do not name a robot, so any robot can
  be spawned in any task; each task's demos come from **one** robot (`meta/robot_key`, checked on every
  episode of the 12 tasks below). Verified in the retarget pipeline so far: task 06 (rigid objects) and 44
  (articulated microwave), target Shadow.
- Hands per arm in the shipped assets:

| Arm | Hands |
|---|---|
| UR5 | Inspire RH56DFX, Inspire RH5DG2, Shadow, Schunk SVH, Wuji |
| xArm7 | Leap, Ability |
| Panda | Orca, Allegro |
| KUKA iiwa7 | Sharpa |
| RM65 | BrainCo Revo2 |
| JAKA Zu7 | DexHand021 |

  Other arm/hand pairs (e.g. UR5 + Leap) need a new asset and would not be official Bench2Dex embodiments.
- Same UR5 arm, but not the same mount: Bench2Dex places the Shadow robot's base at x = 0.75 m, the
  other four UR5 robots at x = 0.5 m (`robots/multi_ur5_*_with_flange.py`). Retargeting matches world-frame
  fingertips, so the arm IK absorbs the offset.

## 2. Scope: the 12 tasks with UR5 demos

| Task | Source hand (teleop) | Target hands (4) |
|---|---|---|
| 06 fruit_bowl_loading | Inspire RH56DFX | RH5DG2, Shadow, Schunk, Wuji |
| 12 screwdriver_box_and_hammer | Inspire RH56DFX | RH5DG2, Shadow, Schunk, Wuji |
| 42 trash_disposal | Inspire RH56DFX | RH5DG2, Shadow, Schunk, Wuji |
| 07 citrus_plate_loading | Inspire RH5DG2 | RH56DFX, Shadow, Schunk, Wuji |
| 34 fridge_wine_interhand_pour | Inspire RH5DG2 | RH56DFX, Shadow, Schunk, Wuji |
| 60 breadbasket_fast_food_loading | Inspire RH5DG2 | RH56DFX, Shadow, Schunk, Wuji |
| 43 fridge_fruit_shelf_sorting | Shadow | RH56DFX, RH5DG2, Schunk, Wuji |
| 76 soup_serving | Shadow | RH56DFX, RH5DG2, Schunk, Wuji |
| 08 frypan_stand_pour | Schunk SVH | RH56DFX, RH5DG2, Shadow, Wuji |
| 44 microwave_bowl_loading | Schunk SVH | RH56DFX, RH5DG2, Shadow, Wuji |
| 21 condiment_box_loading | Wuji | RH56DFX, RH5DG2, Shadow, Schunk |
| 27 ball_box_loading | Wuji | RH56DFX, RH5DG2, Shadow, Schunk |

Robot keys: `multi_ur5_rh56dfx_with_flange`, `multi_ur5_rh5dg2_with_flange`, `multi_ur5_shadow_hand_with_flange`,
`multi_ur5_schunk_hand_with_flange`, `multi_ur5_wuji_with_flange`. All 12 scenes use the same success rule
(MetricTracker stable success, dwell 0.5 s).

## 3. Dataset counts

| Hand | Tasks with original demos | Tasks to retarget | Episodes to generate |
|---|---|---|---|
| Inspire RH56DFX | 3 | 9 | 450 |
| Inspire RH5DG2 | 3 | 9 | 450 |
| Shadow | 2 | 10 | 500 |
| Schunk SVH | 2 | 10 | 500 |
| Wuji | 2 | 10 | 500 |
| **Total** | **12** | **48 (task, hand) pairs** | **2,400** |

Final benchmark: 12 tasks x 5 hands x 50 episodes = **3,000 episodes** (600 original + 2,400 retargeted),
if every episode succeeds; episodes that never pass the success check are left out and listed.

## 4. Pipeline (per episode, per target hand) — `tools/retarget/`

1. Stage 1, reference: replay the source robot kinematically; object poses, source fingertips / wrist / flange,
   fingertip-surface contacts (2 cm), manipulated body per hand.
2. Stage 2, kinematic retarget: whole-body IK of the target robot, fingertip k -> source fingertip k (world frame).
3. Stage 3, SPIDER: MPPI over residual actions with virtual contact guidance annealed to 0. Optimised only when
   manipulation is within 26 frames; residual allowed 13 frames around manipulation; backtracking on drops
   (10, 10, 12 ... 20 frames back). Settings and reasons: `stage3_settings.md`.
4. Stage 4, record: the executed states as a Bench2Dex episode; verdict = MetricTracker stable success (checked
   every step, plus 2 s holding the last command after the demo). Bench2Dex physics unchanged.
5. Stage 5, render: `replay.py --restore-generalization` (state replay, same mode as the public replay data;
   reproduces the run exactly). Open-loop action replay does not, since SPIDER restarts every 5-frame commit
   from a state snapshot.
6. Up to 5 attempts per (episode, target) with new seeds and more samples (`run_target.py`).

## 5. Status

- 06 -> Shadow, prototype code (stopped 10-06): 25/50 episodes run, 14 success
  (`06_fruit_bowl_loading/shadow/legacy_prototype/`). The repo pipeline has not run the full set yet.
- Local data on the dev box (`/workspace/b2d_origin`): 50 episodes for 9 of the 12 tasks; 44 has 45,
  60 and 76 none. A new server downloads them with `tools/retarget/local/setup.sh --fetch`.

## 6. Compute estimate

- Measured (prototype, 06 -> Shadow, ~500-frame episodes, RTX 3090 shared, 2 in parallel): 45-250 min per
  SPIDER attempt, mean 107 min; 56% succeeded on the first attempt.
- 2,400 (episode, target) runs x ~107 min ≈ 4,300 attempt-hours, x ~1.5 for retries ≈ 6,500; at 2 per 24 GB
  GPU that is ~135 days on one RTX 3090, ~17 days on 8 GPUs. Tasks with longer episodes (up to ~1,200 frames)
  cost more; rendering adds ~7 min per successful episode.
- Ways to cut it: more GPUs, fewer samples per step, or a subset first (e.g. 10 episodes per pair).
