# 실물 PiPER 에서 VLA 돌리기 (real_piper)

> 코드 위치: 연구실 PC `~/smolVLA/real_piper` 와 같다 (이 폴더의 `real_piper/`). 다른 PC 에서는 smolVLA 폴더(코드 + 가상환경 + 모델) 안에
> `real_piper/` 로 복사해서, smolVLA 폴더에서 아래 명령을 실행한다. 모델 폴더(`outputs/piper_w12m_model/merged`, 약 1GB)는 저장소에 없다.

시뮬레이터에서 학습한 VLA (SmolVLA + LoRA, w11m·w12m) 를 연구실 실물 PiPER (AgileX, WeGo Robotics 공급) 에 그대로 연결하는 코드다.
시뮬 평가와 **같은 입력, 같은 추론 지연 처리, 같은 동작 변환**을 쓰고, 팔에는 관절 각도로 내보낸다.

> 손목 카메라 마운트 (U20CAM, 3D 프린트): [손목카메라_마운트_U20CAM](손목카메라_마운트_U20CAM/README.md)

> 2026-11-04 기준: 코드 길 전체를 시뮬레이터 팔·카메라에 붙여 시험했다 (아래 "확인한 것"). **실물 팔·카메라에는 아직 한 번도 돌려 보지 않았다.**
> 시뮬 그림으로만 학습한 모델이라, 실물 사진에서는 처음에 잘 안 될 가능성이 크다 ("한계" 절). 이 코드는 그 다음 단계(실물에서 재기·실물 시범 모으기·실물 데이터로 더 학습)를 할 수 있게 하는 바탕이다.

## 어떻게 움직이나

```
위 카메라 ─┐                              ┌─ 동작 묶음 (절대 손끝 목표 x y z, 회전 변화량, 그리퍼)
손목 카메라 ┼→ 시뮬과 같은 관측 → VLA ──────┤
관절 6개 ──┘  (사진 512×512 위아래 뒤집음,  └→ 변화량 = (목표 − 지금 손끝)/0.05 → 목표 손끝·방향 → 역기구학 → 관절 목표 → piper_sdk JointCtrl
그리퍼 거리     손끝·손 방향은 정기구학)                                                    그리퍼: + 닫기 / − 열기 → GripperCtrl
```

- **운동학** (`kin.py`): 시뮬이 쓰는 로봇 + 그리퍼 모델에서 메시만 뺀 `kin/piper_kin.xml`. 정기구학(관절 → 손끝)과 역기구학을 시뮬 제어기
  (`piper_sim/ik_controller.py`) 와 같은 식으로 계산한다. 시뮬도 "변화량 → 역기구학 → 관절 위치" 로 팔을 움직이므로 실물과 길이 같다.
- **관측** (`policy.make_obs`): 손끝 위치 = 그리퍼 가운데 점, 손 방향 = `robot0_right_hand` 몸의 사원수(xyzw), 손가락 = [거리/2, −거리/2].
  좌표는 시뮬 세계 좌표 (로봇 밑동 (0, 0, 0.90), 책상 윗면 z = 0.90).
- **사진** (`cameras.py`): 주점 가운데로 세로 시야각 75° 정사각형을 잘라 512 로 줄이고 위아래를 뒤집는다 (시뮬 관측은 MuJoCo 그림 그대로라 뒤집혀 있다).
- **추론 지연** (`policy.Planner`): 시뮬 평가(`--latency-steps 9 --ttrtc --n-action-steps 1`)와 같은 규칙. 실물에서는 다른 스레드에서 추론하고,
  그동안 지난 계획을 계속 실행하다가 새 계획이 오면 그동안 실행한 만큼 버린다. 추론하는 동안 실행될 앞 동작을 모델에 접두로 넣는다 (Training-Time RTC).
- **물체 위치 입력**: 위 카메라 사진에서 지시문의 물체·놓을 곳을 색으로 찾는다 (`piper_sim/world_detect.py`). 실물은 카메라 보정값과 실물 색 분포가 필요하다.
- **지시**: 영어 틀("put the red cube on the gray square") 또는 한국어 ("빨간 정육면체를 회색 판에 놓아줘", "잠깐, … 먼저", "이어서"). 도는 중에도 쳐서 바꿀 수 있다.
- **다듬기(보조)는 아직 없다.** 시뮬 다듬기는 시뮬 내부 값을 쓰는 곳이 있어 실물로 옮기지 않았다. 지금은 VLA 혼자다.

## 확인한 것 (시뮬레이터에 붙여, 2026-11-04)

| 시험 | 결과 |
|---|---|
| `tests/test_kin_vs_sim.py` 같은 동작에서 관절 목표 (실물용 역기구학 vs 시뮬 제어기) | 차이 0.0000° (60스텝) |
| 같은 시험, 관절 → 손끝 위치·손 사원수 vs 시뮬 관측 | 최대 0.9 mm / 0.002 |
| `tests/test_obs_equiv.py` 같은 순간 VLA 입력·출력 (실물용 관측 vs 시뮬 관측, 같은 잡음) | 상태 입력 차이 대부분 0.006 이하, 동작 차이 대부분 0.02 이하 (그리퍼를 쥐는 순간 0.1 — 실물은 손가락 거리 하나만 알려 줘 두 손가락을 똑같이 나눔) |
| `run_real.py --backend sim` 600스텝 끝까지 | 오류 없음, 추론 13스텝마다 (지연 12) |
| `run_real.py --backend dry --fake-cams` (실제 시간 길: 다른 스레드 추론, 카메라 대신 기준 사진) | 한 스텝 50 ms (20 Hz) 지킴, 예열 뒤 추론 0.7~0.8초 (다른 시험과 GPU 를 나눠 씀) |
| `tests/test_sim_success.py` 여러 장면 성공 수 (실물용 길 vs 시뮬 평가 길, VLA 혼자) | 실물용 길 4/16, 시뮬 평가 길 7/16 (둘 다 시뮬 평가의 VLA 혼자 32% 안팎). 결과가 갈린 3장면(1003·1006·1009)은 추론 잡음 차이 — `tests/test_traj_same_noise.py` 로 잡음을 같게 하면 8장면 모두 결과가 같고 (성공 장면은 스텝 수 1 안팎 차이), 처음 50스텝 손끝 차이 0.03~0.27 cm |

## 순서

### 0. 준비물
- GPU 가 있는 PC (연구실 1080 Ti 로 추론 한 번 0.4~0.6초 — 지연 9~12스텝). `~/smolVLA` 폴더 (코드 + 모델 `outputs/piper_w12m_model/merged`) 와 같은 가상환경
- PiPER + CAN-USB 어댑터, `pip install piper_sdk python-can` (원격 조종·기록은 WeGo 플러그인 `pip install lerobot_robot_wego_piper`). 펌웨어가 S-V1.8-8 이상이면 `pip install "piper_sdk>=0.7.0a1"` 과 설정 `sdk_interface: "v3"`. 설치·CAN·조작법 자세히: LAB-1309 `12_PiPER_조작법_및_설치법`
- 위 카메라: 책상 윗면에서 **0.85 m 위, 로봇 밑동 바로 위**, 똑바로 아래를 보고, 사진 위쪽 = 로봇 앞. 잘라서 **세로 시야각 75°** 가 나와야 한다 (그보다 좁으면 cameras.py 가 멈춤)
- 손목 카메라: 시뮬과 같은 자리 (손 끝판, RealSense D405 크기, 시야각 75° 정사각형) — `reference/wrist_view.png` 와 비슷하게 보이도록
  - U20CAM 이면 3D 프린트 마운트 (0–70 mm·0–100 mm 그리퍼): LAB-1309 `11_실물_PiPER/손목카메라_마운트_U20CAM` — 시뮬 카메라 자리 (J6 축에서 탭 쪽 70 mm) 에 온다. U20CAM 세로 시야는 75° 보다 좁을 수 있다 (그 README 3절)
- 책상: 로봇 밑동을 책상 윗면에 바로 둠 (받침대 없음). 물체·판은 시뮬과 같은 모양·크기·색 (정육면체 4.5 cm, 막대 9×4×4 cm, 원기둥 지름 4.5 cm, 판 9 cm 정사각형 — `piper_sim/worlds/default.json`)

### 1. CAN·팔 점검
```bash
sudo ip link set can0 up type can bitrate 1000000
```
```bash
piper-doctor
```
### 2. 관절이 시뮬 모델과 같은 뜻인지 (팔은 손으로만 움직임)
```bash
python real_piper/check_fk.py --can can0
```
남는 오차가 5 mm 이하여야 다음으로. 크면 `piper-calibrate` (관절 0점) 부터.
그리퍼 방향·벌림 (값 = 벌림, 0 = 닫힘 인지, 정육면체를 쥐었을 때 벌림 → `gripper.q_offset_m`):
```bash
python real_piper/check_gripper.py --can can0
```

### 3. 카메라 달기·보정
- `reference/top_view.png`, `reference/wrist_view.png` (시뮬 기준 사진) 와 실물 화면이 비슷하게 보이도록 단다
```bash
python real_piper/calib_top_camera.py intrinsics --device 0
```
```bash
python real_piper/calib_top_camera.py extrinsics --device 0 --can can0
```
카메라 자리가 시뮬 [0, 0, 1.75] 에서 몇 cm 안, 다시 투영 오차 몇 px 안이면 된다. 결과는 `config_real.json` 에 저장된다.

### 4. 실물 색 맞추기
```bash
python real_piper/fit_colors.py --device 0
```
쓸 물체·판 색을 모두, 책상·로봇도 클릭. 사진 5장 이상 (조명·자리를 바꿔). 결과 `detect_real.json`.

### 5. 연습 (팔에 아무것도 안 보냄)
```bash
python real_piper/run_real.py --backend dry --task "put the red cube on the gray square" --log outputs/real_logs
```
기록(npz)의 사진·손끝 목표가 그럴듯한지 본다.

### 6. 실물 (처음에는 느리게, 비상 정지를 손에)
```bash
python real_piper/run_real.py --backend sdk --speed 20 --task "빨간 정육면체를 회색 판에 놓아줘"
```
처음 자세(시뮬 init_qpos)로 5초에 걸쳐 간 뒤 Enter 를 기다린다. 도는 중 새 지시문을 쳐서 끼어들기를 시험할 수 있다. Ctrl-C → 그 자리에 멈춤.

### 시뮬로 이 코드 길 시험 (팔·카메라 없이)
```bash
PIPER_NOPED=1 MUJOCO_GL=egl python real_piper/run_real.py --backend sim --seed 1000
```

## config_real.json

| 키 | 뜻 |
|---|---|
| `model` | VLA 모델 폴더 (smolVLA 기준 경로) |
| `latency_steps` | 접두로 넣을 지연 (스텝). 실물은 추론 시간에 맞춰 (0.45초 → 9) |
| `can`, `speed_pct` | CAN 이름, 관절 속도 % (MotionCtrl_2) |
| `sdk_interface` | `v2` (C_PiperInterface_V2) 또는 `v3` (펌웨어 S-V1.8-8 이상, piper_sdk 0.7.0a1 이상) |
| `home_qpos` | 처음 자세 (시뮬 init_qpos, rad) |
| `gripper.open_m`, `effort_nm` | 열 때 벌림 (0.07 m), 쥐는 힘 (N·m, 0~5) |
| `gripper.q_offset_m` | 손가락 값 = 거리/2 + 이 값. 시뮬은 4.5 cm 정육면체를 쥐면 두 손가락 합 5.0 cm (패드 두께) → 0.0025 |
| `safety.radius_m` | 손끝 목표를 밑동에서 이 반지름 안으로 자름 (시뮬 작업 공간 0.15~0.45) |
| `safety.z_above_table_m` | 손끝 목표 높이 (책상 위) 아래·위 한계. 아래 한계가 손가락이 책상을 찍지 않게 |
| `safety.max_joint_step_deg` | 한 스텝(0.05초) 관절 이동 한계 |
| `cameras.*` | 장치 번호, 해상도, 내부값(보정), 자르는 세로 시야각, 뒤집기 |
| `top_camera_calib` | 위 카메라 자리·방향 (calib_top_camera.py) — 물체 찾기·로봇 가리개 |
| `detect_model` | 실물 색 분포 (fit_colors.py). 없으면 시뮬 색 분포 (실물에서는 거의 못 찾음) |

## 안전
- 처음은 `--backend dry`, 그다음 `--speed 10~20`, 팔 둘레를 비우고 비상 정지를 손에.
- 손끝 목표는 작업 공간·책상 위 높이로 자르고, 관절은 한 스텝에 4° 이상 못 움직인다 (`safety`). 잘린 횟수는 100스텝마다 출력.
- Ctrl-C 는 지금 관절을 목표로 다시 보내 멈춘다. `EmergencyStop` 은 `robot_io.PiperSDK.estop()` — 팔이 천천히 내려앉으니 아래를 비울 것, 다시 쓰려면 `EmergencyStop(0x02)` 로 풀고 다시 켜는데, 이것은 `ResetPiper` 와 같은 명령이라 힘이 빠져 팔이 떨어진다 (팔이 내려앉은 뒤에 보낼 것). 그래서 실행 중에는 둘 다 쓰지 않는다.

## 한계 (정직하게)
1. **시뮬 그림 → 실물 사진**: VLA 는 MuJoCo 그림으로만 학습했다. 카메라 자리·시야각을 맞춰도 색·빛·질감이 달라 처음에는 잘 안 될 가능성이 크다.
   다음 순서가 현실적이다: ① 이 코드로 실물에서 재기 ② WeGo 플러그인 원격 조종으로 실물 시범 모으기 (`run_real.py` 기록과 같은 꼴로 바꿔 저장)
   ③ 시뮬 + 실물 섞어 더 학습 ④ 사람이 넘겨받는 교정 (HG-DAgger).
2. **다듬기 없음**: 시뮬 성적 81% (일 1개) 는 다듬기를 켠 숫자다. VLA 혼자는 31~33% 다.
3. **관절 0점·방향**: 시뮬 모델(MuJoCo Menagerie, 제조사 URDF 에서 변환)과 실물 펌웨어가 같다고 가정한다. `check_fk.py` 로 먼저 확인.
4. **그리퍼**: 실물은 손가락 사이 거리 하나만 주므로 두 손가락을 똑같이 나눠 넣는다 (시뮬은 쥘 때 0.8 mm 쯤 다르다).
5. **지연**: 실제 추론 시간이 `latency_steps` 와 많이 다르면 접두 길이를 바꿀 것 (실행할 때 출력되는 ms/스텝·추론 횟수로 확인).
