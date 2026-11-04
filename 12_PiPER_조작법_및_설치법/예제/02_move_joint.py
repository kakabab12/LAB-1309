#!/usr/bin/env python3
"""2. 관절 하나 움직이기 — 지금 자세에서 관절 하나만 deg 만큼 돌렸다가 제자리로.

나머지 관절은 지금 각도를 그대로 보낸다. 범위를 넘는 목표는 보내지 않는다.
처음에는 손목 돌리기(J6)를 작게, 속도 10% 로. Ctrl-C 를 누르면 그 자리에 선다 (힘은 빼지 않음).
  python3 02_move_joint.py --joint 6 --deg 10 --speed 10
"""
import time

import common


def main():
    p = common.parser(__doc__.splitlines()[0])
    p.add_argument("--joint", type=int, default=6, choices=range(1, 7), help="움직일 관절 번호 1~6")
    p.add_argument("--deg", type=float, default=10.0, help="돌릴 각도 (°, 30 넘으면 거부)")
    p.add_argument("--speed", type=int, default=10, help="최대 속도의 % (1~30)")
    p.add_argument("--hold", type=float, default=3.0, help="목표마다 보내는 시간 (초)")
    a = p.parse_args()
    if abs(a.deg) > 30 or not 1 <= a.speed <= 30:
        raise SystemExit("예제는 30° 이하, 속도 1~30% 만 받는다")
    piper = common.connect(a.can, a.v3)
    time.sleep(0.2)
    read = common.joints_deg(piper)
    if all(abs(q) < 1e-6 for q in read) and piper.GetCanFps() == 0:
        raise SystemExit("관절 값을 못 읽음 — 01_read_state.py 로 먼저 확인")
    start = common.clip_joints(read)
    for i, (r, s) in enumerate(zip(read, start)):
        if abs(r - s) > 0.5:
            print(f"  J{i + 1} 지금 {r:.1f}° 가 범위 밖 → {s:.1f}° 로 잘라 보냄 (팔이 그만큼 움직임)")
    target = list(start)
    target[a.joint - 1] += a.deg
    lo, hi = common.JOINT_LIMIT_DEG[a.joint - 1]
    if not lo <= target[a.joint - 1] <= hi:
        raise SystemExit(f"J{a.joint} 목표 {target[a.joint - 1]:.1f}° 가 범위 [{lo}, {hi}] 밖 — 반대 방향(--deg {-a.deg:g})으로")
    print(f"지금 관절(°) {common.fmt(start)}")
    print(f"목표 관절(°) {common.fmt(target)}  (J{a.joint} {a.deg:+g}°, 속도 {a.speed}%)")
    common.confirm("팔 둘레를 비웠나? 팔을 켜고 움직인다.")
    common.enable(piper)
    try:
        for name, q in (("목표로", target), ("제자리로", start)):
            print(name)
            t_end = time.time() + a.hold
            while time.time() < t_end:
                common.send_joints_deg(piper, q, a.speed)
                time.sleep(0.01)
            print(f"  지금 관절(°) {common.fmt(common.joints_deg(piper))}")
    except KeyboardInterrupt:
        print("Ctrl-C — 지금 자리에 세움")
    finally:
        here = common.joints_deg(piper)
        for _ in range(20):
            common.send_joints_deg(piper, here, a.speed)
            time.sleep(0.01)
    print("끝. 팔은 켜진 채 그 자리에 있다. 끌 때는 팔을 받치고 (3_조작법.md 9절).")


if __name__ == "__main__":
    main()
