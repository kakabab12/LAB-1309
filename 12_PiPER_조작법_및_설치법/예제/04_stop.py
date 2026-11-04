#!/usr/bin/env python3
"""4. 비상 정지와 풀기.

  python3 04_stop.py            # 비상 정지 (EmergencyStop 0x01): 팔이 천천히 내려앉는다
  python3 04_stop.py --resume   # 풀기 (EmergencyStop 0x02 = ResetPiper): 힘이 바로 빠진다 → 다시 켜기

풀기는 ResetPiper 와 같은 명령이라 힘이 빠져 팔이 떨어진다. 팔이 다 내려앉은 뒤나 손으로 받친 채 쓴다.
공식 demo 설명은 비상 정지 뒤 "reset 하고 enable 을 두 번" 이라고 적었다.
"""
import time

import common


def main():
    p = common.parser(__doc__.splitlines()[0])
    p.add_argument("--resume", action="store_true", help="비상 정지 풀고 다시 켜기")
    a = p.parse_args()
    piper = common.connect(a.can, a.v3)
    if not a.resume:
        piper.EmergencyStop(0x01)
        time.sleep(0.1)
        print(f"비상 정지 보냄. 팔 상태: {piper.GetArmStatus().arm_status.arm_status}")
        print("다시 쓰려면 팔이 내려앉은 뒤: python3 04_stop.py --resume")
        return
    common.confirm("풀면 힘이 바로 빠진다. 팔이 다 내려앉았거나 손으로 받쳤나?")
    piper.EmergencyStop(0x02)
    time.sleep(0.5)
    for i in range(2):                      # demo 설명대로 두 번
        common.enable(piper)
        time.sleep(0.2)
    print(f"다시 켬. 팔 상태: {piper.GetArmStatus().arm_status.arm_status}   관절 켜짐: {piper.GetArmEnableStatus()}")


if __name__ == "__main__":
    main()
