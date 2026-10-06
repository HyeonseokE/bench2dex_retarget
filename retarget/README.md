# retarget — Bench2Dex cross-embodiment SPIDER retargeting

Bench2Dex fork(**[HyeonseokE/Bench2Dex](https://github.com/HyeonseokE/Bench2Dex)**)에 추가한 retargeting 라이브러리다.
레포 하나로 벤치마크 환경(Bench2Dex 원본 코드), retargeting(`retarget/` 라이브러리 + `tools/retarget/` 실행 스크립트),
클러스터 잡(`ondemand/`, 사용법은 [`ondemand/README.md`](../ondemand/README.md))을 함께 관리한다.
저자 원본은 `upstream`(github.com/Bench2Dex/Bench2Dex)으로 연결해 두고, 업데이트는 `git fetch upstream && git merge upstream/main`으로 받는다.

Bench2Dex teleop 데모를 **소스 손에서 나머지 UR5 손 4종으로** 옮긴다. 학습 없이, 태스크 물체의 움직임을
목표 로봇이 Bench2Dex 물리 그대로 재현하도록 SPIDER(MPPI 샘플링 최적화, arXiv 2511.09484)로 다듬는다.
판정은 Bench2Dex 자체 성공 조건(MetricTracker stable success)이다.
`/workspace/retarget`(task 06, RH56DFX → Shadow 프로토타입)을 12개 태스크 × 5종 손으로 일반화했다.

| task | source | targets |
|---|---|---|
| 06 fruit_bowl_loading · 12 screwdriver_box_and_hammer · 42 trash_disposal | RH56DFX | RH5DG2, Shadow, Schunk, Wuji |
| 07 citrus_plate_loading · 34 fridge_wine_interhand_pour · 60 breadbasket_fast_food_loading | RH5DG2 | RH56DFX, Shadow, Schunk, Wuji |
| 43 fridge_fruit_shelf_sorting · 76 soup_serving | Shadow | RH56DFX, RH5DG2, Schunk, Wuji |
| 08 frypan_stand_pour · 44 microwave_bowl_loading | Schunk | RH56DFX, RH5DG2, Shadow, Wuji |
| 21 condiment_box_loading · 27 ball_box_loading | Wuji | RH56DFX, RH5DG2, Shadow, Schunk |

목표 손은 소스를 제외한 4종이 자동으로 정해진다(`TARGETS`로 덮어쓸 수 있음).

## 레이아웃

| 경로 | 내용 |
|---|---|
| `retarget/` | 라이브러리: paths, robots(손 5종 스펙·스포너), task(씬·에피소드 로드), sim_env, spider_env, coupling, geometry, recorder(상태 기록 → Bench2Dex HDF5) |
| `tools/retarget/` | 실행 스크립트: stage1–5, `run_target.py`(시도 반복·status.json·업로드), `upload_hf.py`, `hf_card.py`, `audit_episode.py`, `label_episode.py`, `run_episode.sh` |
| `ondemand/` | SLURM 잡·컨테이너·데이터 다운로드만 (코드 없음) |

출력 루트는 `$B2DR_RUNS`(기본 `$B2D_ROOT/b2dr_runs`), 소스 데이터는 `$B2D_ROOT/b2d_origin/dataset`이다.

## 파이프라인 — 에피소드 하나, 목표 로봇 하나

| stage | 스크립트 | 무엇 | 출력 |
|---|---|---|---|
| 1 | `tools/retarget/stage1_reference.py` | 소스 로봇으로 데모를 **운동학적으로 재생**: 물체 바디 pose, 소스 손끝·손목·플랜지, 손끝-표면 접촉(2 cm), 손별 조작 바디 | `$B2DR_RUNS/<task>/epNNN/reference.npz/.json` |
| 2 | `tools/retarget/stage2_kinematic.py` | 목표 로봇 전신 IK (손끝 k → 소스 손끝 k), 프레임 순서대로 warm-start | `…/<robot>/kinematic.npz` |
| 3 | `tools/retarget/stage3_spider.py` | hold pass 후 SPIDER 최적화. 확정된 실행(env 0)의 **매 물리 스텝 상태**와 매 스텝 관절 목표를 저장(샘플은 저장 안 함, backtracking 시 함께 되감음) | `…/<robot>/<tag>.json`, `<tag>_trace.pkl.gz` |
| 4 | `tools/retarget/stage4_record.py` | 상태 기록을 Bench2Dex DataCollector와 똑같이 기록 (Convention A 액션, box3d, MetricTracker 매 물리 스텝, HDF5EpisodeWriter). **재시뮬레이션 없음**(순수 Python). 판정 = MetricTracker stable success | `$B2DR_RUNS/dataset/<robot>/<scene>/origin-generalization/episode_NNNNNN.hdf5` (성공만) |
| 5 | `tools/retarget/stage5_render.sh` | Bench2Dex `replay.py --restore-generalization --enable-rgb --enable-tactile` (공개 replay 데이터와 같은 `restored` 모드) + GT 라벨(occupancy, box3d/box2d) | `…/replay-generalization/episode_NNNNNN.hdf5` |

stage 4·5의 출력은 HF `Bench2Dex/teleopdata`와 같은 트리·형식이다(`<scene>/origin-generalization`, `<scene>/replay-generalization`).
LeRobot v3가 필요하면 Bench2Dex의 `tools/export/convert_bench2dex_to_lerobot_v3.py`를 그대로 쓴다(ffmpeg, pyarrow 필요).
공개 데이터와 일대일 대조: `python tools/retarget/audit_episode.py --cand <생성 replay> --task_ref <같은 태스크 공개 replay> --robot_ref <목표 손 공개 replay> --source <소스 origin>`.

로컬(dev 박스, 레포 루트에서) 한 에피소드(stage 1–4): `bash tools/retarget/run_episode.sh 06 0 shadow [--num_samples 1024 ...]`

### 설계상 결정

- **운동학은 전부 시뮬레이터 USD에서 읽는다(URDF 미사용).** Schunk URDF는 `thtip` 링크 중복으로 로드되지 않고, RH5DG2에는 손끝 프레임이 없다. FK·Jacobian 모두 PhysX에서 가져오므로 계획과 물리의 운동학이 같다.
- **로봇은 Bench2Dex 스포너를 그대로 쓴다** (`retarget/robots.py:spawn_robot`). 액추에이터 게인, 홈 자세, 드라이브 타입 보정, 손 마찰, 중력보상이 벤치마크와 동일하다. 스포너의 프림 경로 상수만 env_0으로 돌리고 `Articulation`을 N-env 정규식 경로로 감싼다.
- **씬은 Bench2Dex 헬퍼로 만든다** (`retarget/sim_env.py`). 에피소드별 테이블 높이(일반화), 첫 프레임 물체 pose, 관절물체 액추에이터·관절 한계가 반영된다. distractor(clutter)는 시뮬레이션하지 않는다(프로토타입과 동일). 내보낼 때 녹화된 pose를 그대로 복사한다.
- **구동되지 않는 손가락 관절**(RH56DFX·RH5DG2·Schunk의 mimic 관절)은 그 손 자신의 teleop 데모에서 `q = a·q_master + b`로 맞춘다 (`retarget/coupling.py`, `$B2DR_RUNS/coupling/<robot>.json`). 5종 모두 어떤 태스크에서든 소스로 등장하므로 가능하다. 검증 에피소드 오차: RH5DG2 0.025, Schunk 0.024, RH56DFX 0.098 rad(접촉 시 수동 굴곡 때문).
- **성공 판정**은 Bench2Dex `success.evaluate_success_conditions`를 env 0에서 매 프레임 호출한다. `sequence`·`hold_duration`의 상태(ctx)는 스냅샷에 포함되어 backtracking 때 함께 되감긴다.
- **손끝 정의**(`retarget/robots.py`): 순서는 항상 엄지, 검지, 중지, 약지, 소지다. teleop hand cfg의 tip 바디를 쓰고, 바디가 없는 Schunk는 말단 링크 메시에서 가장 먼 점을 쓴다.

## 로컬 GPU 서버 (SLURM·컨테이너 없음)

```bash
# 0) 작업 루트(B2D_ROOT) 아래에 fork를 clone. 이 아래에 IsaacLab, assets, b2d_origin, b2dr_runs, envs/b2d가 생긴다
mkdir -p ~/b2d && cd ~/b2d && git clone https://github.com/HyeonseokE/Bench2Dex.git
echo <HF 토큰> > ~/.hf_token                     # 익명 다운로드는 Hub rate-limit에 걸린다

# 1) 1회 셋업: conda env(py3.11) + torch 2.7(cu128) + Isaac Sim 5.1(pip) + Isaac Lab v2.3.2, 데이터·에셋 다운로드
TASKS="06 12 42 07 34 60 43 76 08 44 21 27" bash Bench2Dex/tools/retarget/local/setup.sh --fetch

# 2) 실행: (task, 에피소드, 목표 손) 큐를 GPU에 나눠 돌린다. 중단 후 다시 실행하면 끝난 것은 건너뛴다
source Bench2Dex/tools/retarget/local/env.sh && cd Bench2Dex
nohup python tools/retarget/run_queue.py --tasks 12 42 --gpus 0 1 --per_gpu 2 --no_upload > ../b2dr_runs/queue.log 2>&1 &

# 3) 진행상황: 2분마다 갱신
cat ../b2dr_runs/STATUS.md          # task x 목표 손별 성공/실패/진행중, 진행 중 에피소드의 마지막 이벤트
cat ../b2dr_runs/results.csv        # (task, 에피소드, 목표)별 상태
```

- 목표 손 기본값은 소스를 뺀 UR5 손 4종이다(`--targets`로 지정 가능). 에피소드마다 stage 1을 한 번 돌리고, 목표마다
  `run_target.py`(stage 2 → SPIDER 최대 `--max_attempts`회 → 기록 → 렌더 → 업로드)를 돌린다. `--no_upload`면 로컬에만 남는다.
- GPU 메모리: SPIDER 1024 샘플 프로세스 하나가 5~6 GB다. 24 GB GPU에서는 `--per_gpu` 3 이하(메모리 부족은 PhysX 상태를 깨뜨린다).
- 헤드리스 컨테이너에서 Isaac Sim이 Vulkan을 못 찾으면: `apt install libvulkan1` 후 `/usr/share/vulkan/icd.d/nvidia_icd.json`을
  `{"file_format_version":"1.0.1","ICD":{"library_path":"libEGL_nvidia.so.0","api_version":"1.4.312"}}`로 둔다.
- 로그: `b2dr_runs/logs/<task>_epNNN_<target>.log`(큐), `b2dr_runs/<scene>/epNNN/<target>/run.log`(단계별).

## 클러스터 (pro6000, OpenOnDemand)

[`ondemand/README.md`](../ondemand/README.md) 참고.

## 검증 현황 (2026-10-05, dev 박스 RTX 3090, 다른 실험과 GPU 공유)

- stage 1, task 06 ep0: 프로토타입(pinocchio FK)과 손끝·손목 위치 차이 0.00 cm, 접촉 판정 일치 99.9%, 조작 구간 ±1 프레임.
- stage 2, 06 ep0 → Shadow: 손끝 오차 평균 0.05 cm(접촉 중 0.09 cm), 최대 관절 점프 0.33 rad(첫 프레임, 프로토타입과 동일).
- stage 3: 강체(06)와 관절물체(44) 모두 끝까지 동작한다. 06 ep0 → Shadow 전체 길이를 **256 샘플**로 돌린 결과는 실패였다
  (프레임 ~300에서 사과를 놓쳤고, 475에서 시간 제한에 걸림). 프로토타입은 같은 에피소드를 1024 샘플로 성공했다.
  기본값(1024)으로 다시 비교해야 한다.
- 이 실행에서 backtracking이 놓친 물체에 프레임마다 재시도를 쓰는 문제가 드러났다. 지금은 조작 구간별로 재시도한다: 그 구간에서 처음 놓친 프레임 기준
  10, 10, 12, 14, 16, 18, 20 프레임 전으로 되돌아가 반복 횟수를 늘려 다시 계획한다(`--back_frames`, `--back_max`). 7회 모두 실패해도 계속 진행한다.
  프로토타입의 성공 에피소드는 한 물체를 6~18번 놓친 뒤에도 다시 잡아 성공한 경우가 많았고, 5~7회에서 중단(`--abort_on_fail 1`)하자 걸린 에피소드(06 ep20~24)가 전부 실패했다.
- 최적화는 조작이 `--gate_look` 26 프레임 안에 있을 때만 하고, residual은 조작 구간 전후 `gate_margin` 13 프레임에서만 허용한다.
- GPU 메모리 부족 등으로 PhysX 상태가 깨지면(물체 pose가 같은 값으로 고정) 결과를 버리고 오류로 끝낸다. 06 ep4에서 가짜 성공이 나온 적이 있다.
- **포맷 e2e** (06 ep0 → Shadow, 짧은 디버그 계획): stage 4 기록 → stage 5 렌더(카메라 6대 JPEG 480×640, TacMap 10 패드,
  occupancy·box 라벨) → Bench2Dex 변환기로 LeRobot v3 생성까지 통과. 공개 replay 파일과 구조 비교 시 남는 차이는
  (a) 손별 tactile 패드 이름(Bench2Dex가 로봇마다 다르게 정의), (b) 지표 스키마 버전(공개 데이터는 이전 Bench2Dex 코드로
  수집, 여기는 fork에 들어 있는 Bench2Dex(fd90dcc 기준)의 MetricTracker)뿐이다. 복원된 씬(배경·텍스처·조명·카메라·distractor)은 공개 RGB와 같은 장면이다.
- **상태 기록 방식인 이유** — 재생에는 두 종류가 있고 결과가 다르다.
  - **state 재생**(기록된 로봇 관절·물체 pose를 매 프레임 그대로 복원, 물리 재계산 없음): 기록 그대로이므로 성공 에피소드는 항상 성공으로 재현된다.
    Bench2Dex `replay.py --restore-generalization`(공개 replay 데이터와 같은 `restored` 모드)이 이 방식이고, stage 4·5가 이를 쓴다.
  - **action 재생**(초기 상태에서 기록된 관절 목표만 넣고 물리를 처음부터 다시 계산): 재현되지 않는다. SPIDER의 확정 실행은 커밋(5 프레임)마다
    스냅샷 복원에서 다시 시작하므로 접촉 캐시·솔버 warm start가 그때마다 초기화되는데, 끊김 없이 다시 계산하면 이 초기화가 없어 작은 차이가 생기고
    아슬아슬한 파지에서 커진다(06 ep3: 프로토타입 자기 env에서도, env 1개·1024개 모두 ~200 프레임에 사과를 떨어뜨림).
  - 기록된 상태 궤적 자체는 Bench2Dex 물리로 시뮬레이션된 결과다(5 프레임마다 같은 상태로의 스냅샷 복원만 끼어 있음). 그래서 실제로 일어난 상태를 기록한다.
- **성공 에피소드 e2e** (프로토타입 SPIDER v2의 06 ep3 → Shadow, 530 프레임): stage 4 기록 → MetricTracker 공식 판정 **성공** →
  stage 5 렌더(390 s)·라벨 → LeRobot v3 변환(528 프레임) 통과. 렌더 영상에서 그릇 이동·사과 2개·바나나 적재가 그대로 보이고
  오른손 파지 구간에 TacMap 신호가 있다.
- 클러스터에서는 아직 한 번도 돌리지 않았다.
