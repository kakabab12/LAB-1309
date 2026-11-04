#!/usr/bin/env python3
"""1. 상태 읽기 — 팔을 움직이지 않는다 (켜지도 않음).

펌웨어 버전, 제어 모드·팔 상태, 관절 각도, 손끝 자세, 그리퍼, 1초 CAN 신호 수를 보여 준다.
펌웨어가 S-V1.8-8 이상이면 이후 예제에 --v3 를 붙인다.
  python3 01_read_state.py            # can0, 2초 동안 0.5초마다
  python3 01_read_state.py --can can_follower1 --seconds 5
"""
import time

import common


def main():
    p = common.parser(__doc__.splitlines()[0])
    p.add_argument("--seconds", type=float, default=2.0)
    a = p.parse_args()
    piper = common.connect(a.can, a.v3)
    time.sleep(0.2)
    fw = piper.GetPiperFirmwareVersion()
    print(f"펌웨어: {fw}" + ("  (읽기 실패 — CAN·전원 확인)" if fw == -0x4AF else ""))
    if isinstance(fw, str) and not a.v3:
        try:
            major, minor = fw.split("V")[1].split("-")[0], fw.split("-")[-1]
            if (int(major.split(".")[0]), int(major.split(".")[1]), int(minor)) >= (1, 8, 8):
                print("  → S-V1.8-8 이상: 다른 예제는 --v3 로 (piper_sdk 0.7.0a1 이상)")
        except (ValueError, IndexError):
            pass
    st = piper.GetArmStatus().arm_status
    print(f"제어 모드: {st.ctrl_mode}   팔 상태: {st.arm_status}   움직임 모드: {st.mode_feed}")
    print(f"관절 켜짐: {piper.GetArmEnableStatus()}")
    t_end = time.time() + a.seconds
    while time.time() < t_end:
        print(f"관절(°) {common.fmt(common.joints_deg(piper))}   손끝(mm, °) {common.fmt(common.end_pose(piper))}"
              f"   그리퍼 {common.gripper_mm(piper):.1f} mm   CAN {piper.GetCanFps()} 개/초")
        time.sleep(0.5)
    print("관절이 모두 0 이고 CAN 신호 수가 0 이면: 팔 전원·CAN 선, 리더 모드인지 확인 (2_CAN_연결.md 5절)")


if __name__ == "__main__":
    main()
