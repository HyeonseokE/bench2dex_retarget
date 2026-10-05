# dex2bench_retarget — Bench2Dex cross-embodiment SPIDER retargeting

Bench2Dex teleop 데모를 **소스 손에서 나머지 UR5 손 4종으로** 옮긴다. 학습 없이, 태스크 물체의 움직임을
목표 로봇이 Bench2Dex 물리 그대로 재현하도록 SPIDER(MPPI 샘플링 최적화, arXiv 2511.09484)로 다듬는다.
판정은 Bench2Dex 자체 성공 조건과 dwell 규칙이다.
`/workspace/retarget`(task 06, RH56DFX → Shadow 프로토타입)을 12개 태스크 × 5종 손으로 일반화한 레포다.

| task | source | targets |
|---|---|---|
| 06 fruit_bowl_loading · 12 screwdriver_box_and_hammer · 42 trash_disposal | RH56DFX | RH5DG2, Shadow, Schunk, Wuji |
| 07 citrus_plate_loading · 34 fridge_wine_interhand_pour · 60 breadbasket_fast_food_loading | RH5DG2 | RH56DFX, Shadow, Schunk, Wuji |
| 43 fridge_fruit_shelf_sorting · 76 soup_serving | Shadow | RH56DFX, RH5DG2, Schunk, Wuji |
| 08 frypan_stand_pour · 44 microwave_bowl_loading | Schunk | RH56DFX, RH5DG2, Shadow, Wuji |
| 21 condiment_box_loading · 27 ball_box_loading | Wuji | RH56DFX, RH5DG2, Shadow, Schunk |

목표 손은 소스를 제외한 4종이 자동으로 정해진다(`TARGETS`로 덮어쓸 수 있음).

## 파이프라인 — 에피소드 하나, 목표 로봇 하나

| stage | 스크립트 | 무엇 | 출력 (`b2dr_runs/<task>/epNNN/`) |
|---|---|---|---|
| 1 | `scripts/stage1_reference.py` | 소스 로봇으로 데모를 **운동학적으로 재생**(프레임별 상태 기록, 물리 스텝 없음): 물체 바디 pose, 소스 손끝·손목·플랜지, 손끝-표면 접촉(2 cm), 손별 조작 중인 바디 | `reference.npz/.json`, `meshes.npz` |
| 2 | `scripts/stage2_kinematic.py` | 목표 로봇 전신 IK: 손끝 k → 소스 손끝 k(접촉 10, 비접촉 2), tool0 플랜지 약하게 유지, 이전 프레임 정칙화. 프레임 순서대로 warm-start | `<robot>/kinematic.npz` |
| 3 | `scripts/stage3_spider.py` | hold pass(물체를 데모 궤적에 고정하고 PD 추종 → 관통 없는 reference) 후 SPIDER: 손목 6-DoF·손가락 residual, 가상 접촉 스프링(반복마다 0으로 감쇠), drop 시 backtracking, settle 40 프레임 | `<robot>/spider.json`, `spider_rollout.npz` |
| 4 | `scripts/stage4_export.py` | 롤아웃을 목표 로봇의 Bench2Dex 에피소드 HDF5로 → `Bench2Dex/replay.py`로 RGB·tactile 렌더 가능 | `<robot>/episode_NNNNNN_<robot>.hdf5` |

로컬(dev 박스)에서 한 에피소드: `bash scripts/run_episode.sh 06 0 shadow [--num_samples 1024 ...]`

### 설계상 결정

- **운동학은 전부 시뮬레이터 USD에서 읽는다(URDF 미사용).** Schunk URDF는 `thtip` 링크 중복으로 로드되지 않고, RH5DG2에는 손끝 프레임이 없다. FK·Jacobian 모두 PhysX에서 가져오므로 계획과 물리의 운동학이 같다.
- **로봇은 Bench2Dex 스포너를 그대로 쓴다** (`b2dr/robots.py:spawn_robot`). 액추에이터 게인, 홈 자세, 드라이브 타입 보정, 손 마찰, 중력보상이 벤치마크와 동일하다. 스포너의 프림 경로 상수만 env_0으로 돌리고 `Articulation`을 N-env 정규식 경로로 감싼다.
- **씬은 Bench2Dex 헬퍼로 만든다** (`b2dr/sim_env.py`). 에피소드별 테이블 높이(일반화), 첫 프레임 물체 pose, 관절물체 액추에이터·관절 한계가 반영된다. distractor(clutter)는 시뮬레이션하지 않는다(프로토타입과 동일). 내보낼 때 녹화된 pose를 그대로 복사한다.
- **구동되지 않는 손가락 관절**(RH56DFX·RH5DG2·Schunk의 mimic 관절)은 그 손 자신의 teleop 데모에서 `q = a·q_master + b`로 맞춘다 (`b2dr/coupling.py`, `b2dr_runs/coupling/<robot>.json`). 5종 모두 어떤 태스크에서든 소스로 등장하므로 가능하다. 검증 에피소드 오차: RH5DG2 0.025, Schunk 0.024, RH56DFX 0.098 rad(접촉 시 수동 굴곡 때문).
- **성공 판정**은 Bench2Dex `success.evaluate_success_conditions`를 env 0에서 매 프레임 호출한다. `sequence`·`hold_duration`의 상태(ctx)는 스냅샷에 포함되어 backtracking 때 함께 되감긴다.
- **손끝 정의**(`b2dr/robots.py`): 순서는 항상 엄지, 검지, 중지, 약지, 소지다. teleop hand cfg의 tip 바디를 쓰고, 바디가 없는 Schunk는 말단 링크 메시에서 가장 먼 점을 쓴다.

## 클러스터 (pro6000, OpenOnDemand)

> **`cluster/main_job.sbatch` 하나만 붙여넣어 던진다.** 나머지는 그 잡이 pull한 체크아웃에서 읽는다.

순서: 토큰 확인 → `$HOME/b2d/dex2bench_retarget` clone/pull → 셋업(Isaac Sim 5.1 SIF, Bench2Dex @ fd90dcc, Isaac Lab v2.3.2;
최초 1회 약 1시간) → 데이터 다운로드 → (task, episode)당 array 태스크 하나 제출. 각 array 태스크가 그 에피소드를 목표 로봇 4종으로
retarget한다. GPU 한 장에 `PACK`(기본 2)개를 동시에 돌리고, `%MAX_GPUS`(기본 2)로 동시 실행 수를 묶는다.

| 파일 | 역할 |
|---|---|
| `cluster/main_job.sbatch` | **진입점(붙여넣는 유일한 파일)** |
| `cluster/retarget_array.sbatch` | array 런처. main_job이 체크아웃 경로에서 제출한다 |
| `cluster/retarget_body.sh` | (task, ep) 하나: stage 1 → 목표별 stage 2/3/4 |
| `cluster/env.sh` · `cluster/setup.sh` | 경로·컨테이너 래퍼 / 1회 셋업 |
| `cluster/fetch_data.py` | HF `Bench2Dex/teleopdata` origin 에피소드 + 선택 태스크의 에셋 |

던지기 전 확인:
1. `~/.hf_token`이 있을 것. 익명 다운로드는 Hub가 중간에 rate-limit한다.
2. 수정한 코드가 GitHub에 push되어 있을 것. `main_job.sbatch` 자신을 고쳤다면 다시 붙여넣는다.

조절 (`sbatch --export=ALL,KNOB=값`):
`TASKS="06 12 ..."` · `EPISODES=10` · `TARGETS="shadow wuji"` · `PACK=2` · `MAX_GPUS=2` · `SPIDER_ARGS="--num_samples 1024 --iters 5"`

**비용.** 기본 범위(12 태스크 × 10 에피소드 × 4 목표)는 SPIDER 480회다. 1회는 에피소드 길이(400~1200 프레임)에 따라 대략
0.5~2 GPU-h로 추정되며, 아직 pro6000에서 실측하지 않았다. GPU 2장이면 여러 날 걸리는 큐다. 첫 결과를 확인한 뒤 `EPISODES`를 늘린다.
모든 단계가 멱등이라 실패하거나 walltime에 걸리면 그냥 다시 던지면 된다.

출력은 `$HOME/b2d/b2dr_runs/<task>/epNNN/<robot>/`에 쌓이고, array 로그는 `slurm-b2dr-retarget-<job>_<idx>.out`이다.

## 검증 현황 (2026-10-05, dev 박스 RTX 3090, 다른 실험과 GPU 공유)

- stage 1, task 06 ep0: 프로토타입(pinocchio FK)과 손끝·손목 위치 차이 0.00 cm, 접촉 판정 일치 99.9%, 조작 구간 ±1 프레임.
- stage 2, 06 ep0 → Shadow: 손끝 오차 평균 0.05 cm(접촉 중 0.09 cm), 최대 관절 점프 0.33 rad(첫 프레임, 프로토타입과 동일).
- stage 3: 강체(06)와 관절물체(44) 모두 끝까지 동작한다. 06 ep0 → Shadow 전체 길이를 **256 샘플**로 돌린 결과는 실패였다
  (프레임 ~300에서 사과를 놓쳤고, 475에서 시간 제한에 걸림). 프로토타입은 같은 에피소드를 1024 샘플로 성공했다.
  기본값(1024)으로 다시 비교해야 한다.
- 이 실행에서 backtracking이 놓친 물체에 프레임마다 재시도를 쓰는 문제가 드러나서 에피소드당 상한(`--max_total_retries 12`)을 넣었다.
- 클러스터에서는 아직 한 번도 돌리지 않았다.
