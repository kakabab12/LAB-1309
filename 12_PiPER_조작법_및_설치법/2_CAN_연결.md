# 2. CAN 연결

PiPER 와 PC 는 CAN 으로 이야기한다. PC 에 USB-CAN 모듈을 꽂으면 `can0` 같은 네트워크 이름이 생긴다. 이것을 1,000,000 bps 로 켜야 SDK 가 쓸 수 있다.

속도는 바꿀 수 없다고 공식 문서가 적었다. 반드시 1000000 이다.

## 1. 잇는 순서

1. 팔 전원을 켠다.
2. 팔의 CAN 선에 이어진 USB-CAN 모듈을 PC USB 에 꽂는다.
3. PC 가 모듈을 알아봤는지 본다.
   ```bash
   ip -br link show type can
   ```
   `can0  DOWN` 같은 줄이 나오면 된 것이다. 아무것도 안 나오면 드라이버를 올리고 다시 본다.
   ```bash
   sudo modprobe gs_usb
   ```
4. CAN 을 켠다 (2절).

## 2. CAN 켜기 — 셋 중 하나

**가. piper_sdk 에 든 스크립트** (0.6.2 에도 있음)

```bash
SDK=$(python3 -c "import piper_sdk, os; print(os.path.dirname(piper_sdk.__file__))")
bash $SDK/can_activate.sh can0 1000000
```

- 연구실 PC 에서는 `python3` 자리에 `~/smolVLA/.venv/bin/python` 을 쓴다.
- 스크립트 안에서 `sudo ip link ...` 를 부르므로 비밀번호를 묻는다.
- can-utils, ethtool 이 없으면 멈추고 설치하라고 알려 준다.

**나. piper_sdk 0.7.0a1 이상**

```bash
piper-can activate can0 1000000
```

**다. 손으로**

```bash
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

## 3. 잘 켜졌는지

```bash
ip -details link show can0      # state UP, bitrate 1000000 이 보이면 됨
candump can0                     # 팔이 보내는 신호가 계속 흘러가면 됨 (Ctrl-C 로 끝)
```

- 팔 하나면 1초에 약 3,000 개 신호가 온다 (공식 Q&A).
- `python3 예제/01_read_state.py` 를 돌리면 1초 신호 수와 펌웨어 버전을 함께 보여 준다.
- CAN 은 재부팅하거나 모듈을 다시 꽂으면 다시 켜야 한다. 자동으로 켜지게 하려면 4절의 WeGo 방법을 쓴다.

## 4. USB-CAN 모듈이 여러 개일 때 (리더 팔 + 팔로워 팔 등)

`can0`, `can1` 은 꽂은 순서대로 붙는 이름이다. 모듈을 다른 USB 구멍에 바꿔 꽂으면 두 팔의 이름이 조용히 뒤바뀐다. 그러면 리더 팔에 팔로워 명령이 갈 수 있다. 그래서 USB 자리나 모듈 일련번호로 이름을 고정한다.

**USB 자리로 (공식 방법)**

1. 모듈 하나만 꽂고 USB 자리를 기록한다.
   ```bash
   sudo ethtool -i can0 | grep bus
   ```
   예: `bus-info: 1-2:1.0`
2. 그 자리의 모듈을 원하는 이름으로 켠다.
   ```bash
   bash $SDK/can_activate.sh can_follower 1000000 1-2:1.0
   ```
3. 다른 모듈은 다른 USB 구멍에 꽂고 같은 방법으로 한다.
4. 둘 이상을 한 번에 켜려면 `$SDK/can_muti_activate.sh` 안의 `USB_PORTS` 와 `EXPECTED_CAN_COUNT` 를 고쳐서 실행한다.

**모듈 일련번호로 (WeGo, 권장)**

```bash
piper-setup                                  # 창에서 팔 이름 정하기 (한 번)
sudo bash scripts/install_can_persist.sh     # lerobot_robot_piper 소스 폴더에서 (한 번)
```

- `piper-setup` 에서 `Find` 를 누르고 그 팔을 손으로 45° 넘게 움직이면, 어느 CAN 이 어느 팔인지 정해진다.
- 그러면 이름이 `can_leader1`, `can_follower1` 처럼 붙고 `~/piper_config.json` 에 저장된다.
- `install_can_persist.sh` 는 모듈 일련번호에 이름을 고정하는 규칙(udev)과, 부팅 때 CAN 을 켜는 서비스를 깐다.
- 이후에는 재부팅하거나 다시 꽂아도 같은 이름으로 켜진다. 지우려면 `sudo bash scripts/install_can_persist.sh uninstall`.
- 어느 모듈이 어느 팔인지 다시 보려면 다음 명령을 쓴다.
  ```bash
  python3 scripts/list_can_adapters.py --watch
  ```
  팔을 손으로 움직이면 해당 줄이 바뀐다.

## 5. 안 될 때

| 증상 | 할 것 |
|---|---|
| `can0` 이 안 보임 | 모듈 USB 다시 꽂기 → `sudo modprobe gs_usb` → `piper-doctor` |
| `ip: command not found` | `sudo apt-get install iproute2` |
| CAN 은 켜졌는데 값이 0 이거나 명령에 반응 없음 | 팔이 리더(마스터) 모드인지 본다. 팔로워로 바꾸고 팔을 다시 켠다 ([3_조작법.md](3_조작법.md) 2절) |
| `SEND_MESSAGE_FAILED (100017)` | 선을 확인하고 팔 전원을 껐다 켠다 (공식 README) |
| 펌웨어 버전이 `-1199` (-0x4AF) | 연결 직후 바로 읽어서다. 0.03초 이상 기다렸다 읽는다 |
| 그리퍼 값이 늘 0 이고 안 움직임 | 처음 쓰는 그리퍼는 벌림 범위 설정이 필요하다 ([3_조작법.md](3_조작법.md) 6절) |

## 다음

→ [3_조작법.md](3_조작법.md)
