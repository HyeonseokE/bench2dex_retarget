# Cross-hand benchmark dataset plan (Bench2Dex, UR5)

Target: a cross-hand (embodiment, not arm) benchmark, made by retargeting Bench2Dex teleop demos onto
the other hands that Bench2Dex mounts on the **same UR5 arm**, so that only the hand differs.

## 1. Bench2Dex at a glance (checked 2026-10-06)

| Item | Paper | Data / code |
|---|---|---|
| Tasks | 26 | 26 task datasets on HF `Bench2Dex/teleopdata`; 27 scene yamls (`86_short_jigsaw_puzzle` has no demos) |
| Robot embodiments | 12 | 12 registered in `robots/__init__.py` |
| Demos | ~1.3K (26 x 50) | per task `origin-generalization/`: `episode_000000..049.hdf5` + the same 50 indices with a `_1` suffix. **Retargeting uses only the 50 files without suffix.** A `_1` file has the same robot joints, actions, object poses and metrics as its original; only the visual randomisation differs (iTHOR background, table texture, lighting). `replay-generalization/` is the rendered version of both and is not needed either (checked on 06 ep0/7/21, 2026-10-06) |

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

## 2b. Hand set for the cross-hand dataset (decided 2026-10-12)

Six hands. Five share the UR5 arm, so only the hand differs between them; LEAP adds a hand with a different
finger count (it is an official Bench2Dex embodiment only on xArm7). DoF counted per hand from the Bench2Dex
robot models (joints) and teleop configs (`teleop/hand_cfgs/*.yml`, driven joints = what retargeting plans).

| Hand | Arm | Fingers | DoF (joints / driven) | Key trait | Why it is in the set |
|---|---|---|---|---|---|
| Inspire RH56DFX | UR5 | 5 | 12 / 6 | linkage-coupled, low DoF (6 mimic joints) | the most common commercial humanoid hand; low-DoF anchor |
| Inspire RH5DG2 | UR5 | 5 | 18 / 13 | same family, different shape and fingertips (5 passive joints) | near pair to RH56DFX: effect of a small shape change |
| Schunk SVH | UR5 | 5 | 20 / 9 | 9 actuators drive 20 joints through couplings | mid DoF; far fewer drives than joints |
| Wuji | UR5 | 5 | 20 / 20 | every joint driven | high DoF, direct drive |
| Shadow | UR5 | 5 | 24 / 22 | human-sized, most joints (teleop fixes THJ3, LFJ5) | high-DoF anthropomorphic; hardest to retarget |
| LEAP | xArm7 | 4 | 16 / 13 | four fingers, individually driven (teleop fixes 3 abduction joints) | different finger count |

Comparisons the set supports (same source episode, scene and objects for every hand):

1. DoF axis (RH56DFX, RH5DG2 -> Schunk -> Wuji, Shadow): does more driven DoF make retargeting and policy
   learning easier or harder (success rate, attempts, policy success).
2. Near-pair transfer (RH56DFX <-> RH5DG2): the easiest cross-hand generalisation case.
3. Same DoF class, different build (Wuji <-> Shadow): separates DoF from shape and size.
4. Finger count (five-finger hands <-> LEAP): how grasps change with one finger fewer, and whether five-finger
   data generalises to four fingers. Arm differs here (xArm7), so this axis is not hand-only.
5. Retargeting difficulty already measured on task 07 (first-attempt success): RH56DFX 94 %, Wuji 76 %,
   Schunk 60 %, Shadow 48 %.

Extra source tasks (demos not on UR5): 79 bimanual_piano_melody (xArm7 + Ability) and 80 gaming_desk_setup
(JAKA ZU7 + DexHand021), 50 episodes each (indices 0-49), targets = the six hands above. Not in the six,
not used: Ability and DexHand021 as targets, Panda (Allegro, Orca), RM65 (Revo2), iiwa7 (Sharpa).

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
