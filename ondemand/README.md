# ondemand — SLURM jobs (pro6000, OpenOnDemand)

이 폴더는 **클러스터 실행만** 담당한다: 잡 스크립트, 컨테이너·경로 설정, 1회 셋업, 데이터 다운로드.
retargeting 구현은 레포의 `retarget/`(라이브러리)와 `tools/retarget/`(실행 스크립트)에 있다 — [`retarget/README.md`](../retarget/README.md).

## 붙여넣는 파일은 둘뿐

OOD Job Composer → New Job → 파일 내용 붙여넣기 → Submit.

| 순서 | 파일 | 무엇 |
|---|---|---|
| 1 | `env_check.sbatch` | 환경 점검. L1 CUDA · L2 apptainer/SIF · L3 Isaac Sim · L4 카메라 · L5–L10 파이프라인 1–5단계(06 ep0, 짧게) · L11 HF 쓰기 권한. 출력은 `$HOME/b2d/b2dr_runs_envcheck/` |
| 2 | `main_job.sbatch` | 실제 작업. 태스크 하나 × 50 에피소드 → 소스 외 4종 손으로 retarget → 기록·렌더 → HF 업로드 |

나머지 파일은 잡이 clone/pull한 fork 체크아웃(`$HOME/b2d/bench2dex_retarget`)에서 읽는다. 그래서 `retarget/`, `tools/retarget/`,
`ondemand/*.sh`를 고친 경우에는 push만 하면 되고, 진입점 sbatch 자체를 고쳤을 때만 다시 붙여넣는다.

## 흐름 (`main_job.sbatch`)

1. preflight — `~/.hf_token` 확인 (익명 다운로드는 Hub rate-limit에 걸린다)
2. git sync — `$HOME/b2d/bench2dex_retarget`에 fork clone 또는 fast-forward (예전 저자 원본 체크아웃이 있으면 fork로 전환)
3. setup — Isaac Sim 5.1 SIF, Isaac Lab v2.3.2 (`setup.sh`, 최초 1회 약 1시간)
4. fetch — teleop 에피소드 + 선택 태스크 에셋 (`fetch_data.py`, 있는 건 건너뜀)
5. submit — (task, episode)당 array 태스크 하나 (`retarget_array.sbatch` → `retarget_body.sh` → `tools/retarget/run_target.py`),
   array가 끝나면 `finalize.sbatch`가 각 HF 레포의 README와 failures.json을 쓴다

각 에피소드는 목표 손별로 MetricTracker 성공이 나올 때까지 `MAX_ATTEMPTS`번 시도(시드 변경)하고, 성공하면 바로 push한다.
끝내 실패한 에피소드는 빠지고 `failures.json`에 남는다. 모든 단계가 멱등이라 다시 던져도 끝난 단계와 이미 올린 에피소드는 건너뛴다.

## 조절 (`sbatch --export=ALL,KNOB=값` 또는 sbatch 상단 기본값 수정)

| knob | 기본 | 의미 |
|---|---|---|
| `TASKS` | `06` | 제출당 태스크 하나 권장 (06 12 42 07 34 60 43 76 08 44 21 27) |
| `EPISODES` | `50` | 소스 에피소드 수 |
| `TARGETS` | 소스 외 4종 | 예: `"shadow wuji"` |
| `MAX_ATTEMPTS` | `5` | (에피소드, 목표)당 SPIDER 시도 횟수 |
| `PACK` / `MAX_GPUS` | `2` / `2` | GPU당 동시 목표 수 / 동시 GPU 수 |
| `SPIDER_ARGS` | 없음 | 예: `"--num_samples 1024 --iters 5"` |
| `UPLOAD` · `HF_NAMESPACE` · `HF_STAGES` | `1` · 토큰 사용자 · `origin replay` | 업로드 설정 |

## HF 출력

레포 `<ns>/b2d-<scene>-<target>-retargeting` (public), 예: `b2d-06_fruit_bowl_loading-shadow-retargeting`

```
dataset/<scene>/<target>/origin-generalization/episode_NNNNNN.hdf5   # 상태·액션·지표 (렌더 없음)
dataset/<scene>/<target>/replay-generalization/episode_NNNNNN.hdf5   # origin + RGB 6대·TacMap·라벨
```

## 파일

| 파일 | 역할 |
|---|---|
| `env.sh` | 경로(`$HOME/b2d` → 컨테이너 `/workspace`), apptainer 래퍼(`ISAAC`), 토큰, `sync_repo` |
| `setup.sh` | SIF 빌드, Isaac Lab 설치, python 의존성 (1회) |
| `fetch_data.py` | HF `Bench2Dex/teleopdata` origin 에피소드 + coupling용 에피소드 + 에셋 |
| `retarget_array.sbatch` / `retarget_body.sh` | array 런처 / (task, ep) 하나: stage 1 → 목표별 `run_target.py` |
| `finalize.sbatch` | 레포 카드·failures.json |
| `checks/` | `env_check.sbatch`가 쓰는 L1/L3/L4 점검 스크립트 |

로그: `slurm-b2dr-{envcheck,main}-<job>.out`, `slurm-b2dr-retarget-<job>_<idx>.out`, `slurm-b2dr-finalize-<job>.out`.
작업 출력: `$HOME/b2d/bench2dex_retarget/results/<scene>/epNNN/<robot>/`.

**비용 미실측.** SPIDER 1회는 에피소드 길이(400~1200 프레임)에 따라 대략 0.5~2 GPU-h로 추정한다. 클러스터에서는 아직 돌려 보지 않았으니
`env_check.sbatch` → `main_job.sbatch`(TASKS=06) 순서로 시작한다.
