# ondemand/bench2dex — Bench2Dex retargeting SLURM jobs (pro6000, OpenOnDemand)

Bench2Dex 텔레옵 데모를 다른 손으로 SPIDER retargeting하는 클러스터 잡. 이 폴더는 잡·컨테이너·셋업·다운로드만 담고,
retargeting 코드는 레포의 `retarget/`(라이브러리)와 `tools/retarget/`(실행 스크립트)에 있다 — [`retarget/README.md`](../../retarget/README.md).
(2026-10-12 기준)

## 붙여넣는 파일 (OOD Job Composer → New Job → 내용 붙여넣기 → Submit)

| 파일 | 무엇 |
|---|---|
| `main_job.sbatch` | **실제 작업.** pro6000 2장을 처음부터 잡고 한 잡에서 다운로드 → retargeting → HF 업로드 판정. 태스크 하나당 잡 하나, `TASK_QUEUE` 순서로 다음 태스크 잡을 자동으로 이어 붙임 |
| `upload_job.sbatch` | 끝난 태스크를 지금 업로드(기본 부분 업로드). 나중에 빠진 에피소드만 채워 올리면 HF에 있는 것과 합쳐 50/50 카드로 갱신 |
| `sysinfo.sbatch` | 노드 사양(드라이버, CUDA, OS, SLURM, apptainer, 컨테이너)과 Isaac Sim 렌더링 확인 → `SPEC.md` |
| `env_check.sbatch` | 파이프라인 단계별 점검(L1 CUDA ~ L11 HF 권한) |
| `cleanup_tmp.sbatch` | 노드 `/tmp`의 사용자 잔여물 정리 |

나머지 파일은 잡이 pull한 fork 체크아웃(`$HOME/b2d/bench2dex_retarget`)에서 읽는다. 그래서 이 폴더의 `*.sh`, `retarget/`,
`tools/retarget/`를 고치면 push만 하면 되고, 붙여넣는 sbatch 자체를 고쳤을 때만 다시 붙여넣는다.

## `main_job.sbatch` 흐름

1. preflight — `~/.hf_token` 확인
2. git sync — fork를 clone 또는 fast-forward. 다음 태스크 잡(`afterany`)을 이 시점에 대기열에 넣음
3. setup — Isaac Sim 5.1 SIF, Isaac Lab v2.3.2 (`setup.sh`, 최초 1회 약 1시간)
4. fetch — 태스크의 소스 에피소드 50개, 커플링용 에피소드, 에셋 일부만 (`fetch_data.py`): 쓰는 로봇(UR5 손 5종, xArm7+LEAP,
   79·80 소스), 12개 UR5 태스크와 현재 태스크의 물체, `scenes/` (약 2 GB). HF 429(요청 한도)는 기다렸다가 이어 받음
5. retarget — `retarget_worker.sbatch`: CUDA MPS, GPU당 `PER_GPU`(10) 세션 = 20개 동시. (에피소드, 목표 손) 큐를
   `tools/retarget/run_queue.py` → `run_target.py`로 처리 (stage 1~4: reference, 기구학 IK, SPIDER v2, Bench2Dex HDF5 기록).
   렌더링은 하지 않음
6. upload — `finalize.sbatch`: 목표 손별로 50개 모두 성공했을 때만 HDF5(origin)와 데이터셋 카드를 올림

- 시도: (에피소드, 손)당 최대 `MAX_ATTEMPTS`(3)회. 시도 K: 시드 K, 샘플 1024·(1+K//2), iters 5+K (`experiments/stage3_settings.md`)
- 다시 제출하면 끝난 (에피소드, 손)은 건너뜀. 실패한 건을 다시 돌리려면 `MAX_ATTEMPTS`를 늘린다
- 같은 태스크를 돌리는 잡이 이미 실행 중이면 그 뒤로 대기열에 다시 들어가고 끝남(태스크별 잠금 `~/b2d/locks/`).
  태스크가 다르면 동시에 돈다(`PARALLEL=0`이면 한 번에 하나)

## 조절 (sbatch 상단 기본값 수정, 또는 `--export=ALL,KNOB=값`)

| knob | 기본 | 의미 |
|---|---|---|
| `TASK_QUEUE` | `79 80` | 이 순서로 태스크당 잡 하나씩 |
| `TASK_TARGETS` | `79:` 6종, `80:` 6종 | 태스크별 목표 손 (`"79:shadow,wuji 80:leap"`). 없는 태스크는 기본 목록 |
| `TARGETS` | 없음 | 모든 태스크에 같은 목표 손 (`TASK_TARGETS`보다 우선) |
| `TASKS` | 큐의 첫 태스크 | `TASKS=07`만 주면 그 태스크 하나 (체인 없음) |
| `EPISODES` | `50` | 소스 에피소드 0..N-1 |
| `MAX_ATTEMPTS` | `3` | (에피소드, 손)당 SPIDER 시도 횟수 |
| `PER_GPU` · `MPS_ON` | `10` · `1` | GPU당 세션 수 · MPS |
| `PARALLEL` | `1` | 다른 태스크 잡과 동시 실행 허용 |
| `UPLOAD` · `HF_NAMESPACE` · `HF_STAGES` | `1` · 토큰 사용자 · `origin` | 업로드 |

기본 목표 손: UR5 태스크는 소스를 뺀 UR5 손 4종, 다른 팔로 녹화된 태스크(79, 80)는 6종(UR5 5종 + xArm7+LEAP).
손 선정 이유와 DoF는 `experiments/benchmark_plan.md` 2b절.

## 진행 확인 (클러스터 셸 없음 → 잡이 스스로 보고)

- `~/b2d/b2dr_status_task<태스크>.md` (OOD Files, 5분마다): 손별 성공·실패·진행, 시도별 성공(첫 시도/재시도), 최근 큐 줄, 오류,
  GPU 메모리·세션 수·MPS·노드 `/tmp`
- 잡 로그 `slurm-b2dr-main-<id>.out`: 단계 진행, `[monitor ...]` 요약 줄, 마지막 업로드 결과
- 잡 폴더의 `OUTPUTS.md`: 결과 파일 경로와 손별 HDF5 개수
- 에피소드별 진행: `results/report/<scene>/<target>/STATUS.md` (`last event`에 frame 진행)

## 결과와 HF

- 클러스터 저장: `$HOME/b2d/bench2dex_retarget/results/dataset/<target>/<scene>/origin-generalization/episode_NNNNNN.hdf5`
- HF 레포: `<ns>/b2d-<scene>-<target>-retargeting` (public), 경로 `dataset/<scene>/<target>/origin-generalization/`
- 이미지 없음(origin). RGB·촉각이 필요하면 RTX 렌더가 되는 곳에서 `tools/retarget/stage5_render.sh`

## 파일

| 파일 | 역할 |
|---|---|
| `env.sh` | 경로(`$HOME/b2d` → 컨테이너 `/workspace`), apptainer 래퍼(`ISAAC`), 토큰, `sync_repo` |
| `setup.sh` | SIF 빌드, Isaac Lab 설치, python 의존성 (1회) |
| `fetch_data.py` | 소스 에피소드 + 커플링 에피소드 + 필요한 에셋만 |
| `retarget_worker.sbatch` | 2 GPU 워커: MPS 시작, GPU id 확인, `run_queue.py`, 끝에 손별 50/50 여부 |
| `finalize.sbatch` | 손별 업로드 판정 |
| `monitor.sh` · `outputs_note.sh` | 상태 파일 · `OUTPUTS.md` |
| `checks/` | `sysinfo`·`env_check`가 쓰는 CUDA, Kit, 카메라 점검 스크립트 |

## 알려진 사항

- 노드 p1: RTX PRO 6000 ×4, 드라이버 580.82.07 (Isaac Sim 5.1 테스트 계열). 상세: `experiments/ENVIRONMENT.md`
- 카메라 렌더링 점검(R2/L4c)은 2026-10-07에 20분 타임아웃이었으나, 그때 점검 스크립트가 `--camera` 없이 호출되어 카메라를
  실제로 시험하지 않았다(2026-10-12 수정). 클러스터 렌더링 가능 여부는 `sysinfo.sbatch`로 다시 확인 필요
- 노드 `/tmp`(루트 디스크)는 99% 사용 중. Isaac 캐시와 MPS 파일이 쓰므로 상태 파일의 `/tmp` 여유를 확인
- `ondemand/` 바로 아래의 `env.sh` 등은 2026-10-12 이전에 제출된 잡을 위한 `bench2dex/` 심볼릭 링크다. 그 잡들이 끝나면 지운다
