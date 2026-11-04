#!/usr/bin/env python3
"""3. 그리퍼 — 벌렸다(open-mm) 닫았다 다시 벌린다.

손가락 사이에 아무것도 두지 않는다. 힘은 기본 1 N·m (최대 5).
값이 늘 0 이고 안 움직이면 처음 쓰는 그리퍼라 벌림 범위 설정이 필요하다 (3_조작법.md 6절).
  python3 03_gripper.py --open-mm 50
"""
import time

import common


def move(piper, mm, effort_nm, seconds):
    t_end = time.time() + seconds
    while time.time() < t_end:
        piper.GripperCtrl(int(round(mm * 1000)), int(round(effort_nm * 1000)), 0x01, 0)   # 벌림 0.001 mm, 힘 0.001 N·m, 켬
        time.sleep(0.01)
    print(f"  목표 {mm:.0f} mm → 지금 {common.gripper_mm(piper):.1f} mm")


def main():
    p = common.parser(__doc__.splitlines()[0])
    p.add_argument("--open-mm", type=float, default=50.0)
    p.add_argument("--effort", type=float, default=1.0, help="N·m (0~5)")
    a = p.parse_args()
    if not 0 < a.open_mm <= common.GRIPPER_MAX_MM or not 0 < a.effort <= 5:
        raise SystemExit(f"벌림은 0~{common.GRIPPER_MAX_MM:g} mm, 힘은 0~5 N·m")
    piper = common.connect(a.can, a.v3)
    print(f"지금 그리퍼 {common.gripper_mm(piper):.1f} mm")
    common.confirm("손가락 사이를 비웠나? 팔을 켜고 그리퍼를 움직인다.")
    common.enable(piper)
    piper.GripperCtrl(0, int(a.effort * 1000), 0x02, 0)     # 오류 지우기 (끄고 지움)
    time.sleep(0.05)
    for mm in (a.open_mm, 0.0, a.open_mm):
        move(piper, mm, a.effort, 2.0)
    print("끝 (벌린 채). 팔은 켜진 채 그 자리에 있다.")


if __name__ == "__main__":
    main()
