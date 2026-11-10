# Bench2Dex VR 텔레옵 — Meta Quest 3 + CloudXR 6 (isaacteleop)

Manus 장갑 / iPhone ARKit / 페달 없이 **Quest 3 하나로** 양손 UR5(+손) 데모를 수집한다.
DexVerse `teleop/quest-cloudxr6` 브랜치와 같은 스택(Quest 브라우저 WebXR → CloudXR 6.1 런타임 →
Isaac Sim OpenXR)을 쓰고, Bench2Dex 쪽 리타게팅(DexPilot)·양팔 IK·collector는 그대로 쓴다.

## 구성

```
[Quest 3 브라우저: IsaacTeleop client] ──Tailscale──▶ [이 컨테이너 (tailscaled userspace)]
     양손 26 joint, START/STOP/RESET  ⇄  화면 스트리밍       ├ CloudXR 6.1 런타임 (env isaacteleop, WSS 48322)
                                                              └ main.py --teleop-device quest (Isaac Sim, XR kit)

main.py --teleop-device quest
  └ teleop/xr_hand_source.py   XrHandSource: Isaac Lab OpenXRDevice (retargeter 없이 raw joint)
  └ teleop/xr_teleop.py        XrTeleopController: 21 keypoint + 손목 프레임 → RetargetBridge (DexPilot, hand_cfgs)
                               XrIsaacLabBridge: 손목 상대 매핑 → 양팔 ArmIKController
  └ collector (HDF5) / homing / 씬 재샘플 — 기존 코드 그대로
```

손목 매핑(DexVerse `quat_absolute` 모드와 같은 방식):
anchor 시점의 사람 손목 `(p_h0, R_h0)`, 로봇 손바닥 `(p_r0, R_ee0)` 기준으로
`palm 목표 = p_r0 + s·(p_h − p_h0)`, `EE 회전 = (R_h·R_h0ᵀ)·R_ee0`. 회전은 쿼터니언으로 IK에
들어가므로 오일러 짐벌락 / ±180° 점프가 없다.

anchor(= clutch)가 다시 잡히는 때:
- homing이 끝날 때마다 (헤드셋 START/STOP/RESET 후),
- 손이 0.3 s 넘게 추적 안 되다가 다시 보일 때 → **손을 시야 밖으로 뺐다가 옮겨서 다시 넣으면 로봇은 안 튄다**,
- 한 번에 8 cm 넘게 튀는 프레임이 5번 연속일 때.

## 진행 순서

### 1회 설치

```bash
cd /workspace/bench2dex_teleop          # 브랜치 teleop/quest-cloudxr6 의 worktree
bash tools/teleop_xr/setup_tailscale.sh # tailscale 설치 + tailscaled(userspace) + `tailscale up`
                                        # → 출력된 로그인 URL을 열어 이 노드(b2d-teleop)를 tailnet에 승인
bash tools/teleop_xr/setup_isaacteleop_env.sh   # /workspace/envs/isaacteleop + isaacteleop[cloudxr]==1.0.193
```

- 이 컨테이너는 `/dev/net/tun`이 없고 docker bridge 네트워크(172.17.x)라서 tailscaled를 userspace 모드로 띄운다.
  tailnet 피어(Quest)가 100.x 주소로 들어오면 컨테이너 안의 로컬 리스너로 전달된다.
- 루트 overlay가 거의 꽉 차 있어서 CloudXR 상태 폴더 `~/.cloudxr`는 `/workspace/.cloudxr` 심볼릭 링크다.
- 컨테이너를 재시작하면 `setup_tailscale.sh`만 다시 실행하면 된다(로그인 상태는 /workspace에 남는다).

### 매번 실행

```bash
# 터미널 1 — CloudXR 런타임 (계속 띄워 둠). 최초 1회 NVIDIA EULA [y/N] → y
tools/teleop_xr/start_cloudxr_runtime.sh
# "CloudXR runtime: running / WSS proxy: running" 이 뜨면 OK

# 터미널 2 — 텔레옵 (먼저 녹화 없이 조작감 확인)
tools/teleop_xr/run_quest_teleop.sh --task scenes/06_fruit_bowl_loading.yaml
# 데모 녹화
tools/teleop_xr/run_quest_teleop.sh --task scenes/06_fruit_bowl_loading.yaml --collect
```

`run_quest_teleop.sh`가 붙이는 기본값(직접 주면 그 값): `--teleop --teleop-device quest --headless --xr-stream-log 100`.
`--gui`를 주면 headless를 빼고 화면을 띄운다.

### Quest 3

1. Tailscale 앱 켜고 같은 tailnet에 로그인. `tools/teleop_xr/ts status`에 quest가 보이는지 확인. 서버 IP는 `tools/teleop_xr/ts ip -4`.
2. 인증서 수락: Quest 브라우저에서 `https://<서버 tailscale IP>:48322` → 고급 → 계속 ("Certificate Accepted").
3. `https://nvidia.github.io/IsaacTeleop/client` → Server IP `<서버 tailscale IP>`, Port `48322` → Connect → Enter VR.
4. 손추적: `chrome://flags`의 WebXR 실험 기능 ON(브라우저 재시작), 설정에서 손 추적 ON·컨트롤러 OFF, 사이트 손추적 권한 허용.
5. 양손이 보이는 상태에서 시작한다(손이 처음 보이는 순간이 anchor).

### 녹화 조작 (`--collect`)

| 헤드셋 메뉴 | 동작 |
|---|---|
| **START** | 로봇 home 복귀 → 정확한 HOME으로 스냅 → 녹화 시작. 이때 양손 anchor가 새로 잡힌다 |
| **STOP** | home 복귀 후 에피소드 저장 → 씬 재샘플 |
| **RESET** | 현재 에피소드 폐기 + home |

- home 복귀 중에는 손을 따라가지 않는다. 다음 녹화를 위해 편한 자세로 손을 두고 기다리면 된다.
- `--xr-auto-stop-success-steps N`: 녹화 중 태스크 성공이 N 스텝 유지되면 자동 STOP(저장).
- GUI 실행이면 기존 키보드(NUMPAD 1/2/3)도 같이 동작한다.

### RGB 넣기 (녹화 후)

XR 세션 안에서 collector 카메라를 렌더링하면 첫 프레임에서 멈춘다(DexVerse에서 확인). 그래서
`--teleop-device quest`는 `--enable_cameras`를 거부하고, 녹화는 state만 한다(기본 collect config는 rgb/depth off).
영상은 녹화 후 kinematic state replay로 넣는다(같은 서버에서):

```bash
source /workspace/bench2dex_env.sh
python replay.py --hdf5 <episode.hdf5> --enable-rgb --headless
```

## 조정 옵션

| 옵션 | 기본값 | 내용 |
|---|---|---|
| `--xr-anchor-pos x y z` | `0 -0.55 0` | 헤드셋 바닥 원점에 보일 sim 좌표. 로봇 몸통 바로 뒤에서 테이블을 내려다보는 위치 |
| `--xr-anchor-rot w x y z` | `1 0 0 0` | 단위 회전 = 작업자가 sim +Y(로봇 정면) 방향을 봄. 방향이 틀리면 z축 회전으로 조정 |
| `--xr-pos-scale` | `1.0` | 사람 손목 이동량 → 로봇 손바닥 이동량 |
| `--xr-wrist-smoothing` | `0.5` | 손목 자세 저역통과(0=원본, 클수록 부드럽고 느림). 20 Hz 폴링 기준 |
| `--xr-stream-log N` | `0` (스크립트 100) | N 폴링마다 손 스트림 요약 `[xr] ... nonzero=26/26`과 추종 오차 `[xr-track]` 출력 |
| `--xr-replay file.npz` | – | 헤드셋 대신 녹화된 손 데이터 재생(XR 세션 없음). 아래 참고 |

손가락 크기: `teleop/hand_cfgs/<hand>.yml`의 `xr_hand_scale_xyz` / `xr_hand_offset_xyz` (Quest 전용;
Manus용 `hand_scale_xyz`/`hand_offset_xyz`는 그대로 둔다). 값은 성인 평균 손(손목→중지 끝 약 19 cm) 기준으로
편 손 손목→손끝 길이가 로봇과 같게 맞춘 것이다(rh56dfx 1.14, rh5dg2 1.17, shadow 1.10, schunk_svh 1.11, wuji 1.10).
실행 중 YAML을 고치면 hot-reload된다. 손이 작은 작업자는 값을 키운다.

## 헤드셋 없이 테스트

```bash
source /workspace/bench2dex_env.sh
# 1) 손가락 경로 (손 5종 × 양손: 손끝 벡터 오차, 주먹 시 손가락 닫힘, 손 전체 자세 불변성)
python teleop/tests/test_xr_retarget.py
# 2) 실제 Quest 손 데이터 녹화: 텔레옵 중 B2D_XR_DUMP=/path/hands.npz 를 주면 매 폴링 raw joint 저장
B2D_XR_DUMP=/workspace/output/hands.npz tools/teleop_xr/run_quest_teleop.sh --task ...
# 3) 녹화한(또는 합성한) 손으로 전체 루프(리타게팅 + 양팔 IK + collector) 재생
python main.py --task scenes/06_fruit_bowl_loading.yaml --headless --teleop --teleop-device quest \
    --xr-replay /workspace/output/hands.npz --xr-stream-log 40
```

검증 결과(합성 손, task 06, multi_ur5_wuji):
- 손가락: 5종 × 양손 손끝 벡터 오차 0.1–2.4 cm(Schunk 주먹만 4 cm, 로봇이 끝까지 못 쥠), 주먹 시 손끝 길이 42–60%로 감소,
  손 전체를 임의 회전/이동해도 손끝 변화 ≤ 0.2 mm.
- 손목: 전완 회전 ±70°, 굽힘 ±45°, 편위 ±25°를 줘도 손바닥 회전 오차 평균 1–3°, 최대 3.5°(1 업데이트 지연 포함),
  위치 오차 ≤ 0.6 cm. 씬 재샘플 후에도 다시 anchor되고 계속 추종.

## 트러블슈팅

| 증상 | 확인 |
|---|---|
| `[xr] ... nonzero=0/26` 계속 | 런타임 로그(`/workspace/.cloudxr/logs/`)의 `Selected devices`가 Push Hand Tracker면 런타임 재시작(스크립트가 push=0 설정). Quest WebXR flag / 손추적 / 권한 확인 |
| `LOST(n)` | 손이 추적 범위 밖. 로봇 팔은 마지막 목표 유지, 0.3 s 이상이면 다시 보일 때 anchor 재설정 |
| Quest "WebSocket timeout" | `https://<IP>:48322` 인증서를 다시 수락. `/workspace/.cloudxr/logs/wss.*.log`에 Quest IP의 `Proxying` 줄이 없으면 요청이 서버에 안 온 것(Tailscale/네트워크) |
| 로그인 후 약 15 s 뒤 끊김 | 미디어 endpoint 문제. 런타임 시작 로그의 `media endpoint advertised`가 tailscale IP인지 확인 |
| userspace Tailscale로 스트림이 안 붙음 | 대안: 호스트에 Tailscale + 컨테이너를 `--network host`로 재생성(DexVerse work1 구성). 그때는 `CXR_ENDPOINT_IP` 없이 호스트 IP가 쓰인다 |
| `CloudXR runtime is not running` | 터미널 1의 런타임을 먼저 띄울 것 |
| 손목 방향이 반대로 움직임 | `--xr-anchor-rot`가 작업자 방향과 안 맞음(단위 회전 = sim +Y를 바라봄) |

## 구현 메모

- **OpenXR → 21 keypoint**: palm과 검지~새끼 metacarpal 4개를 빼고 엄지 metacarpal은 남긴다(MediaPipe 21과 같은 구성, DexVerse와 동일 인덱스).
- **손목 프레임**: RetargetBridge가 기대하는 프레임(Manus 손목 노드에서 만들던 것)은 OpenXR wrist joint 축으로
  `(+Y, −X, +Z)`, 즉 joint 프레임을 Z축으로 +90° 돌린 것이며 양손 동일하다. UR5 손 5종의 zero-pose 손끝(모두 손가락 +z,
  검지 쪽 +y로 같은 규약)에 합성 손을 정렬해서 구했다(잔차 < 6°). 다른 후보 프레임은 손끝 오차가 6–30 cm로 커진다.
- 기존 Manus 경로(`--teleop-device manus`, 기본값)는 동작이 바뀌지 않는다(`RetargetBridge.retarget`이 새
  `retarget_keypoints`를 호출하도록 분리만 함).
- 실제 Quest를 연결한 세션은 아직 이 서버에서 검증하지 않았다(합성/재생 데이터로만 검증).
