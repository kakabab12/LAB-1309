# 물체 — 로봇팔·실험 물체 STL 모음

나중에 SO-101 로봇팔을 다시 구매하거나 출력해서 실물 실험을 할 때 필요한 3D 프린팅 파일 모음.

| 폴더 | 내용 | 출처 |
|---|---|---|
| [로봇팔_SO101/](로봇팔_SO101/) | SO-101 팔로워·리더 팔 부품, 손목·상단 카메라 마운트, 출력 정확도 게이지 | TheRobotStudio/SO-ARM100 공식 저장소 (Apache-2.0) |
| [실험물체/](실험물체/) | 적재 실험의 분류 통, 큐브 (시뮬레이션과 같은 크기) | 이 연구에서 생성 (`act_sim/make_object_stl.py`) |

## 1. 로봇팔 SO-101 (`로봇팔_SO101/`)

**키트를 살 경우**: 출력된 부품이 포함된 키트(Seeed Studio, WowRobo 등, [README_원본.md](로봇팔_SO101/README_원본.md) 의 Kits)를 사면 팔 부품은 출력할 필요가 없고, **카메라 마운트(아래 ③)와 실험물체만** 출력하면 된다.

**직접 출력할 경우** (공식 권장: PLA+, 노즐 0.4mm·층 0.2mm, 채움 15%, 서포트 사용):

| | 파일 | 용도 |
|---|---|---|
| ① 한 판에 모은 파일 | `STL/SO101/Follower/Prusa_Follower_SO101.stl` · `BambuLabA1mini_Follower_SO101.stl` | **팔로워(로봇) 팔** 전체 부품, 프린터 판 크기에 맞춰 배치됨 |
| | `STL/SO101/Leader/Prusa_Leader_SO101.stl` · `BambuLabA1mini_Leader_SO101.stl` | **리더(조종) 팔** 전체 부품 — 사람 시연(원격조작)을 하려면 필요 |
| ② 부품별 파일 `STL/SO101/Individual/` | 공통: Base, Base_motor_holder, Motor_holder_Base, Motor_holder_Wrist, Under_arm, Upper_arm, Rotation_Pitch, Wrist_Roll_Pitch, WaveShare(또는 Seeedstudio)_Mounting_Plate | 팔로워·리더 둘 다 1세트씩 |
| | 팔로워 전용: Moving_Jaw, Wrist_Roll_Follower | 그리퍼 |
| | 리더 전용: Handle, Trigger, Wrist_Roll | 손잡이 |
| ③ 카메라 마운트 `Optional/` | `Wrist_Cam_Mount_32x32_UVC_Module/stl/..._SO101.stl` | **손목 카메라** (32×32mm USB 카메라 모듈용, 팔로워의 Wrist_Roll_Follower 대신 장착) |
| | `Overhead_Cam_Mount_32x32_UVC_Module/stl/` (4개) | **상단(오버헤드) 카메라** 거치대 |
| ④ 게이지 `STL/Gauges/` | Gauge_0 · Gauge_tight_1 (서보 기준), Lego 시험편 | 본 출력 전에 프린터 치수 정확도 확인 |

- 이 연구의 ACT 는 **손목 + 상단 카메라 2개**를 쓰므로 ③ 두 가지가 모두 필요하다 (시뮬레이션은 공식 `so101_new_calib_camera.xml` 손목 카메라 모델 사용).
- 팀이 썼던 카메라 모델명은 기록이 없어 32×32mm USB 카메라 모듈 기준 마운트를 넣었다. 다른 카메라(RealSense D405/D435, 일반 웹캠)용 마운트는 공식 저장소 `Optional/` 폴더에 있다.
- 조립 순서·배선은 [LeRobot SO-101 문서](https://huggingface.co/docs/lerobot/so101) 참고.

**모터·전자부품 (공식 README 기준, 출력물 아님)**

| 부품 | 팔로워 1대 | 리더 1대 |
|---|---|---|
| STS3215 서보 7.4V | 1/345 기어(C001) × 6 | C046(1/147) × 3, C044(1/191) × 2, C001(1/345) × 1 |
| 모터 제어 보드 (Waveshare) | 1 | 1 |
| 전원 5V (12V 서보를 쓰면 12V 5A 이상), USB-C 케이블, 책상 고정 클램프 | 1 | 1 |

## 2. 실험 물체 (`실험물체/`)

시뮬레이션(`act_sim/stack_env.py`)과 **같은 크기**로 만든 파일. 단위 mm, 닫힌(watertight) 메시로 검사함.

| 파일 | 크기 | 용도 |
|---|---|---|
| `bin_64x64x52mm_wall2mm.stl` | 바깥 64 × 64 × 52mm, 벽·바닥 두께 2mm, 윗면 열림 | 분류 통 (저울 위 → 1층 / 옆 / 2층 적재). **파란색 PLA** 로 출력 (색은 판정 기준이라 바꾸지 않음) |
| `cube_18mm.stl` | 18 × 18 × 18mm | 컨베이어가 통에 떨어뜨리는 큐브 |

<img src="실험물체/bin_preview.png" width="240">

- 통 무게: 시뮬레이션은 33g. 꽉 채워 출력하면 약 41g(PLA)이므로 채움 비율로 맞추거나, 실제 무게를 재서 큐브 수로 **118g 기준**을 맞춘다 (시뮬레이션: 통 33g + 큐브 10g × 9개 = 123g).
- 2층 적재는 위 통이 아래 통의 2mm 벽 위에 얹히므로, 벽 윗면이 평평하게 나오도록 출력 방향은 **바닥이 아래**로 둔다.
- 저울은 구매품(디지털 저울, 7-segment 표시), 컨베이어는 MakerWorld 의 *Automatic Color Sorting Conveyor* (Brian3D) — 라이선스상 파일을 여기 다시 올리지 않고 원본에서 받는다.

## 출처·라이선스

- `로봇팔_SO101/` : [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) 커밋 `a758567` (2026-10-09 받음) 의 일부 그대로. **Apache License 2.0** — [LICENSE](로봇팔_SO101/LICENSE) 원문 동봉. 큰 조립 확인용 파일(`SO101 Assembly.stl`, 38MB)과 Ender 용 판 파일은 용량 때문에 제외 (원본 저장소에서 받을 수 있음).
- `실험물체/` : 이 연구에서 생성. 다시 만들려면 `python make_object_stl.py --out 실험물체` (act_sim).
