# ondemand — SLURM jobs (pro6000, OpenOnDemand)

이 폴더는 **클러스터 실행만** 담당한다: 잡 스크립트, 컨테이너·경로 설정, 1회 셋업, 데이터 다운로드.
retargeting 구현은 레포의 `retarget/`(라이브러리)와 `tools/retarget/`(실행 스크립트)에 있다 — [`retarget/README.md`](../retarget/README.md).

## 붙여넣는 파일은 둘뿐

OOD Job Composer → New Job → 파일 내용 붙여넣기 → Submit.

| 순서 | 파일 | 무엇 |
|---|---|---|
| 1 | `env_check.sbatch` | 환경 점검. L1 CUDA · L2 apptainer/SIF · L3 Isaac Sim · L4 카메라 · L5–L10 파이프라인 1–5단계(06 ep0, 짧게) · L11 HF 쓰기 권한. 출력은 `$HOME/b2d/b2dr_runs_envcheck/` |
| 0 | `sysinfo.sbatch` | 노드 소프트웨어 사양(드라이버·CUDA·OS·SLURM·apptainer·컨테이너) + Isaac Sim 렌더링 확인. 결과 `SPEC.md` → `experiments/ENVIRONMENT.md`에 정리 |
| 2 | `main_job.sbatch` | 실제 작업. 태스크 하나 × 50 에피소드 → 소스 외 4종 손으로 SPIDER(v2) retarget → Bench2Dex HDF5 기록 → 손별로 50개 모두 성공했을 때만 HF 업로드 (렌더링 없음) |

나머지 파일은 잡이 clone/pull한 fork 체크아웃(`$HOME/b2d/bench2dex_retarget`)에서 읽는다. 그래서 `retarget/`, `tools/retarget/`,
`ondemand/*.sh`를 고친 경우에는 push만 하면 되고, 진입점 sbatch 자체를 고쳤을 때만 다시 붙여넣는다.

## 흐름 (`main_job.sbatch`)

1. preflight — `~/.hf_token` 확인 (익명 다운로드는 Hub rate-limit에 걸린다)
2. git sync — `$HOME/b2d/bench2dex_retarget`에 fork clone 또는 fast-forward (예전 저자 원본 체크아웃이 있으면 fork로 전환)
3. setup — Isaac Sim 5.1 SIF, Isaac Lab v2.3.2 (`setup.sh`, 최초 1회 약 1시간)
4. fetch — teleop 에피소드 + 선택 태스크 에셋 (`fetch_data.py`, 있는 건 건너뜀)
5. submit — 워커 잡 하나(`retarget_worker.sbatch`): pro6000 2장, CUDA MPS, GPU당 `PER_GPU`(10)개 세션 = 20개 동시.
   (에피소드, 목표 손) 큐를 `tools/retarget/run_queue.py` → `run_target.py`로 처리하고(에피소드마다 stage 1 한 번),
   끝나면 `finalize.sbatch`가 `tools/retarget/upload_task.py`를 실행한다

각 에피소드는 목표 손별로 MetricTracker 성공이 나올 때까지 `MAX_ATTEMPTS`번 시도한다(로컬과 같은 v2 기본값·시도 스케줄,
`experiments/stage3_settings.md`). 클러스터에서는 RTX 렌더링을 하지 않고 기록된 HDF5(origin)만 만든다.
업로드는 **목표 손별로 소스 50개가 모두 성공했을 때만** 한 번에(50개 + 데이터셋 카드) 한다. 하나라도 빠지면 그 손은 올리지 않고
빠진 에피소드를 로그에 출력한다. 모든 단계가 멱등이라 다시 던지면 성공한 에피소드는 건너뛰고 실패한 것만 다시 시도한 뒤 다시 업로드를 판정한다.

## 진행 확인

로그인 노드(OOD 셸)에서 `bash ~/b2d/bench2dex_retarget/ondemand/status.sh`. 잡 목록, (태스크, 손)별 성공·실패·진행 수
(`results/report/STATUS.md`, 2분마다 갱신), 워커 로그의 최근 시작·종료와 오류, 노드의 GPU 메모리·세션 수·MPS 상태·`/tmp` 여유를 보여 준다.

## 조절 (`sbatch --export=ALL,KNOB=값` 또는 sbatch 상단 기본값 수정)

| knob | 기본 | 의미 |
|---|---|---|
| `TASKS` | `07` | 제출당 태스크 하나 권장 (06 12 42 07 34 60 43 76 08 44 21 27) |
| `EPISODES` | `50` | 소스 에피소드 수 |
| `TARGETS` | 소스 외 4종 | 예: `"shadow wuji"` |
| `MAX_ATTEMPTS` | `5` | (에피소드, 목표)당 SPIDER 시도 횟수 |
| `PER_GPU` | `10` | GPU당 동시 세션 수 (2장 → 20개) |
| `MPS_ON` | `1` | CUDA MPS 데몬을 잡 안에서 켬 (노드에 `nvidia-cuda-mps-control`이 없으면 경고 후 MPS 없이 실행) |
| `SPIDER_ARGS` | 없음 (= v2) | stage 3 추가 인자. 비워 두면 v2 기본값 |
| `UPLOAD` · `HF_NAMESPACE` · `HF_STAGES` | `1` · 토큰 사용자 · `origin` | 업로드 설정 (`UPLOAD=0`이면 finalize 생략) |

## HF 출력

레포 `<ns>/b2d-<scene>-<target>-retargeting` (public), 예: `b2d-06_fruit_bowl_loading-shadow-retargeting`

```
dataset/<scene>/<target>/origin-generalization/episode_NNNNNN.hdf5   # 상태·액션·지표 (렌더 없음) -- 클러스터 기본
dataset/<scene>/<target>/replay-generalization/episode_NNNNNN.hdf5   # origin + RGB 6대·TacMap·라벨 -- 렌더링이 되는 곳에서만 (클러스터 아님)
```

## 파일

| 파일 | 역할 |
|---|---|
| `env.sh` | 경로(`$HOME/b2d` → 컨테이너 `/workspace`), apptainer 래퍼(`ISAAC`), 토큰, `sync_repo` |
| `setup.sh` | SIF 빌드, Isaac Lab 설치, python 의존성 (1회) |
| `fetch_data.py` | HF `Bench2Dex/teleopdata` origin 에피소드 + coupling용 에피소드 + 에셋 |
| `retarget_worker.sbatch` | 2 GPU 워커: MPS 시작, GPU id 확인, `run_queue.py` 실행, 끝에 손별 50/50 여부 출력 |
| `status.sh` | 진행 확인 (제출하지 않고 로그인 노드에서 실행) |
| `finalize.sbatch` | 손별 업로드 판정(50/50일 때만 HDF5 + 카드 업로드) |
| `checks/` | `env_check.sbatch`가 쓰는 L1/L3/L4 점검 스크립트 |

로그: `slurm-b2dr-{envcheck,main}-<job>.out`, `slurm-b2dr-retarget-<job>_<idx>.out`, `slurm-b2dr-finalize-<job>.out`.
작업 출력: `$HOME/b2d/bench2dex_retarget/results/<scene>/epNNN/<robot>/`.

**비용 미실측.** SPIDER 1회는 에피소드 길이(400~1200 프레임)에 따라 대략 0.5~2 GPU-h로 추정한다. 클러스터에서는 아직 돌려 보지 않았으니
`env_check.sbatch` → `main_job.sbatch`(TASKS=06) 순서로 시작한다.
