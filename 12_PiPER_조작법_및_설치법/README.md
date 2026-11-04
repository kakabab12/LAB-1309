# PiPER 조작법 및 설치법

> 2026-11-04 정리. 연구실 PiPER(AgileX 제품, WeGo Robotics 공급)를 PC 에서 움직이는 데 필요한 설치, 연결, 조작 방법이다.
> 공식 자료를 읽고 정리했다: agilexrobotics/piper_sdk (10/14 판), WeGo-Robotics/lerobot_robot_piper (0.3.1).
> 설치와 명령 확인은 연구실 PC(Ubuntu 22.04, Python 3.10)에서 직접 해 봤다. **팔을 연결한 시험은 아직 안 했다.**

## 들어 있는 것

| 파일 | 내용 |
|---|---|
| [1_설치법.md](1_설치법.md) | CAN 도구, piper_sdk, WeGo 플러그인 설치와 확인 |
| [2_CAN_연결.md](2_CAN_연결.md) | 팔과 PC 를 잇고 CAN 을 켜는 법, 잘 됐는지 보는 법, 안 될 때 |
| [3_조작법.md](3_조작법.md) | 켜기, 관절 움직이기, 그리퍼, 손끝 자세, 상태 읽기, 멈추기, 끄기 |
| [4_원격조종과_기록.md](4_원격조종과_기록.md) | 리더 팔로 팔로워 팔 움직이기, LeRobot 으로 시범 기록 |
| [예제/](예제/README.md) | 바로 돌려 볼 수 있는 짧은 파이썬 파일 4개 (상태 읽기, 관절, 그리퍼, 멈추기) |

## 용어

| 말 | 뜻 |
|---|---|
| SDK (piper_sdk) | 제조사가 준 파이썬 라이브러리. 팔에 명령을 보내고 상태를 읽는다 |
| CAN | 자동차·로봇에서 쓰는 통신선. PiPER 는 USB-CAN 모듈로 PC 와 잇고, PC 에서는 `can0` 같은 이름으로 보인다 |
| 리더 팔 (master, 티칭 팔) | 사람이 손으로 움직이는 팔 |
| 팔로워 팔 (slave) | 명령대로 움직이는 팔. **SDK 로 움직이려면 팔로워 모드여야 한다** |
| 켜기 (enable) / 끄기 (disable) | 모터에 힘을 넣기 / 빼기. 끄면 힘이 바로 빠져 팔이 떨어진다 |
| 펌웨어 | 팔 안의 제어 프로그램 버전 (예: S-V1.6-3). 버전에 따라 쓰는 SDK 판이 다르다 |
| WeGo 플러그인 | WeGo 가 만든 LeRobot 연결 도구. 원격 조종, 시범 기록, 설정 창(GUI)을 준다 |

## 한눈에 보는 순서

1. **설치 (한 번)**: CAN 도구 → piper_sdk → (원격 조종·기록을 하려면) WeGo 플러그인
   ```bash
   sudo apt update && sudo apt install can-utils ethtool iproute2
   pip3 install python-can piper_sdk
   ```
2. **연결**: 팔 전원 켜기 → USB-CAN 모듈을 PC 에 꽂기 → CAN 켜기
   ```bash
   bash "$(python3 -c 'import piper_sdk, os; print(os.path.dirname(piper_sdk.__file__))')/can_activate.sh" can0 1000000
   ```
3. **확인 (팔은 안 움직임)**: 펌웨어 버전과 관절 각도가 나오면 된 것
   ```bash
   python3 예제/01_read_state.py
   ```
4. **움직여 보기**: 관절 6(손목 돌리기)을 10° 돌렸다가 돌아오기, 속도 10%
   ```bash
   python3 예제/02_move_joint.py --joint 6 --deg 10 --speed 10
   ```
5. **끝낼 때**: 팔을 접은 자세로 옮기고, 손으로 받친 채 전원을 끈다. SDK 로 끄면 힘이 빠져 떨어진다.

## 연구실 PC 에서 확인한 것 (11/4)

| 확인 | 결과 |
|---|---|
| `pip install piper_sdk` 로 받아지는 판 | **0.6.2**. V2 만 있고 `piper-can` 명령이 없다. 펌웨어 S-V1.8-8 이상용 V3 는 미리 보기판 0.7.0a1 에 있다 |
| 연구 가상환경(`~/smolVLA/.venv`)에 piper_sdk 를 넣으면 | 새로 깔리는 것은 piper-sdk 0.6.2, python-can 4.6.1, wrapt 1.17.3 셋뿐. 다른 것은 안 바뀐다 (`--dry-run` 으로 확인) |
| WeGo 플러그인 0.3.1 + LeRobot 0.4.4 | `piper-doctor` 1단계 통과: LeRobot 이 `piper_follower`·`piper_leader` 를 자동으로 찾는다. 2단계부터는 팔을 안 꽂아서 실패 (정상) |
| WeGo 플러그인을 의존성째 깔면 | opencv-python 5.0 이 새로 깔려, 이미 있는 opencv-python-headless 4.12 와 `cv2` 가 겹친다. 그래서 연구 가상환경에는 `--no-deps` 로 넣는다 |
| `lerobot-teleoperate`, `lerobot-record` 에 PiPER 넣기 | 명령과 설정은 그대로 읽힌다. CAN 에 연결하는 단계에서 멈춤 (팔이 없어서) |
| 그림 창 도구 (`piper-ui` 등) | 연구 가상환경 파이썬에 tkinter 가 없다. `sudo apt install python3-tk` 가 필요하다 |
| CAN 도구 | can-utils, ethtool 이 아직 안 깔려 있다 (sudo 필요). USB-CAN 드라이버(gs_usb)는 커널에 있다 |
| 예제 4개 | 가짜 SDK 를 붙여 끝까지 돌려서 순서와 단위 계산을 확인했다. 실물 팔로는 아직 안 돌려 봤다 |

## 안전 (꼭 읽을 것)

- 처음에는 속도 10% 로 돌린다. 팔이 닿는 곳을 비우고, 한 사람은 전원 스위치 옆에 있는다.
- **힘이 빠지는 명령 세 가지**는 팔을 손으로 받치거나 내려앉은 뒤에만 쓴다. 셋 다 팔이 떨어진다.
  - `DisablePiper()` (끄기)
  - `ResetPiper()` (리셋)
  - `EmergencyStop(0x02)` (비상 정지 풀기). `ResetPiper` 와 같은 CAN 명령이다.
- `EmergencyStop(0x01)` (비상 정지)를 보내면 팔이 천천히 내려앉는다. 아래를 비워 둔다.
- 관절 범위 밖 목표는 보내지 않는다 ([3_조작법.md](3_조작법.md) 3절).
- 영점 설정(`piper-calibrate`, `JointConfig(set_zero)`)은 모터 안에 저장되어 되돌리기 어렵다. 필요할 때만 한다.
- MIT 모드(관절 힘 직접 제어)는 쓰지 않는다. 공식 문서도 잘못 쓰면 팔이 망가진다고 경고한다.

## 출처

- piper_sdk: <https://github.com/agilexrobotics/piper_sdk> (커밋 b847e53, 2026-10-14)
  - README.MD, assets/can_config.MD, assets/Q&A.MD, assets/double_piper.MD, assets/V3/CHANGELOG-V3.MD
  - piper_sdk/demo/V2, V3
- WeGo 플러그인: <https://github.com/WeGo-Robotics/lerobot_robot_piper> (커밋 ed387ed, 0.3.1)
- 시뮬에서 학습한 VLA 를 실물에 잇는 우리 코드: [11_실물_PiPER](../11_실물_PiPER/README.md)
