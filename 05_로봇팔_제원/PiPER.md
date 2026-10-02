# AgileX PiPER 제원

연구실에 있는 실물 로봇팔이다. 앞으로의 실험 기준 로봇이다 (2026-10-03 결정).

## 구입처와 연결 소프트웨어

| 항목 | 내용 |
|---|---|
| 제조 | AgileX Robotics |
| 연구실 장비 공급 | **WeGo Robotics** (위고 로보틱스) |
| LeRobot 연결 플러그인 | [WeGo-Robotics/lerobot_robot_piper](https://github.com/WeGo-Robotics/lerobot_robot_piper) (Apache-2.0) |
| 설치 | `pip install lerobot_robot_wego_piper` — PyPI의 `lerobot_robot_piper`는 다른 회사 것이라 주의 |
| 요구 사항 | Python 3.10 이상, LeRobot 0.3.0 이상, Linux SocketCAN, CAN-USB 어댑터(gs_usb) |
| 원격 조종 | 리더 팔을 손으로 움직이면 팔로워 팔이 따라 함 (`piper-teleop`) |
| 데이터 기록 | 관절 위치 + USB 카메라 여러 대 (LeRobot 데이터셋) |
| 점검·설정 | `piper-doctor`(설치·CAN·팔 응답 점검), `piper-setup`(팔 이름), `piper-calibrate`(영점), `piper-ui` |
| 안전 | `max_relative_target` 로 한 스텝 관절 이동 제한 |

주의: 이 플러그인은 **관절 각도**로 기록하고 움직인다. 시뮬레이터의 우리 모델은 **손끝 위치·방향 변화량**으로 움직인다. 실물 단계에서 정기구학(관절 → 손끝)과 역기구학(손끝 → 관절) 변환을 넣거나, 관절 각도로 학습하는 방식을 정해야 한다.

## 공식 사양 (판매 페이지)

출처: [Generation Robots PiPER](https://www.generationrobots.com/en/404258-6-axis-robotic-arm-piper.html), [US Robot Store PiPER](https://www.usrobotstore.com/products/piper)

| 항목 | 값 |
|---|---|
| 자유도 | 6 (+ 그리퍼) |
| 가반하중 | 1.5 kg |
| 도달 거리 | 626 mm |
| 무게 | 4.2 kg |
| 반복 정밀도 | ±0.1 mm |
| 전원 | DC 24 V |
| 통신 | CAN (1 Mbps, USB-CAN 어댑터) |
| 재질 | 알루미늄 몸체, 플라스틱 덮개 |
| 제어기 | 팔에 내장 |
| 프로그래밍 | 손으로 끌어 가르치기, 경로 입력, API (piper_sdk), ROS |

## 관절 범위와 크기 (제조사 공식 URDF)

출처: AgileX [Piper_ros](https://github.com/agilexrobotics/Piper_ros) 저장소 `noetic` 브랜치 `src/piper_description/urdf/piper_description.urdf` (2026-10-03 직접 읽음)

| 관절 | 범위 | 비고 |
|---|---|---|
| J1 (밑동 회전) | −150° ~ +150° | |
| J2 (어깨) | 0° ~ +180° | |
| J3 (팔꿈치) | −170° ~ 0° | |
| J4 (손목 회전) | −100° ~ +100° | |
| J5 (손목 굽힘) | −70° ~ +70° | |
| J6 (손 회전) | −120° ~ +120° | |
| 그리퍼 손가락 2개 | 각 0 ~ 35 mm (최대 약 70 mm 벌어짐) | 직선 운동 |

| 마디 | 길이 |
|---|---|
| 밑동 → 어깨 높이 | 123 mm |
| 위팔 (J2 → J3) | 285 mm |
| 아래팔 (J3 → J4·J5) | 약 252 mm |
| 손목 → 손 끝판 (J5 → J6) | 91 mm |
| 손 끝판 → 손가락 | 136 mm |
| URDF 질량 합계 | 4.67 kg |

URDF에 적힌 관절 힘 한계는 100 (그리퍼 10), 속도 한계는 5 rad/s (J6 3 rad/s). 실제 모터 사양과 다를 수 있다.

### 자료마다 다른 값 (주의)

| 출처 | J3 | J4 | J6 |
|---|---|---|---|
| 공식 URDF (`noetic`) | −170° ~ 0° | ±100° | ±120° |
| MuJoCo Menagerie 모델 (`ros-noetic-no-aloha` URDF에서 변환) | −154.5° ~ 0° | ±105° | ±180° |
| roboticscenter.ai 사양표 | −150° ~ +80° | ±175° | ±175° |
| Generation Robots | | | ±170° |

실물에서 쓰기 전에 팔의 펌웨어 한계를 `piper_sdk`로 직접 읽어 확인해야 한다.

## 시뮬레이터 모델

- MuJoCo Menagerie `agilex_piper` (MIT 라이선스, Google DeepMind 관리) — `third_party/mujoco_menagerie/agilex_piper` 에 받아 둠
- 관절 6개 위치 제어 + 손가락 2개 (오른쪽이 왼쪽을 따라감)
- 손끝 기준점: 손 끝판(link6)에서 접근 방향으로 12 cm (손가락 패드 사이)

## LIBERO-Goal 장면에서 손이 닿는지 (2026-10-03 계산)

PiPER 밑동을 책상 위(높이 0.90 m, y=0)에 놓고, 그리퍼가 아래를 보게 해서 역기구학으로 계산했다. 위치 오차 1 cm, 방향 오차 5° 안이면 "닿음".

| 밑동 x 위치 | 닿지 않는 곳 |
|---|---|
| −0.55 m (책상 가장자리 밖) | 거의 다 |
| −0.45 m | 접시 위, 서랍 손잡이, 캐비닛 윗면, 서랍 안, 와인 선반, 스토브 앞 |
| −0.35 m | 접시 위(2 cm 모자람), 서랍 손잡이, 캐비닛 윗면, 서랍 안, 와인 선반 |
| −0.30 m | 와인병 위, 서랍 손잡이, 캐비닛 윗면, 서랍 안, 와인 선반 |

LIBERO의 캐비닛(손잡이·윗면이 책상 위 19~23 cm)과 와인 선반은 PiPER에게 높고 멀다. 그리퍼가 아래를 보려면 손목이 그보다 21 cm 위에 있어야 하는데, 어깨에서 손목까지 최대 약 54 cm(285 + 252 mm)라 모자란다. 그래서 PiPER 장면은 받침대 높이·위치를 바꾸거나 캐비닛·선반을 가깝고 낮게 배치해야 한다. 실물 책상도 같은 배치로 맞춘다.
