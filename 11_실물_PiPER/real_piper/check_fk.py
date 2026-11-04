#!/usr/bin/env python
"""실물 관절 각도와 시뮬 모델이 같은 뜻인지 확인 (팔은 움직이지 않음, 2026-11-04).

팔을 손으로 끌어(가르치기 모드) 또는 piper-ui 로 서로 다른 자세 8~10개에 둘 때마다 Enter.
자세마다 SDK 가 계산한 손 끝판 위치(GetArmEndPoseMsgs)와 우리 모델(kin/piper_kin.xml)로 계산한 link6 위치를 모아,
'link6 좌표계에서 고정된 차이 o' 하나로 모두 설명되는지 본다 (p_sdk = p_link6 − 밑동 + R_link6 · o).
  남는 오차가 몇 mm 이하 → 관절 0점·방향·단위가 시뮬 모델과 같다 (run_real.py 써도 됨)
  수 cm 이상 → 관절 0점(piper-calibrate)·관절 부호·펌웨어 DH 오프셋(dh_is_offset)을 먼저 맞출 것
  python real_piper/check_fk.py --can can0
"""
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_piper.kin import Kin  # noqa: E402
from real_piper.robot_io import PiperSDK  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--can", default="can0")
    p.add_argument("--n", type=int, default=8)
    a = p.parse_args()
    kin = Kin()
    link6 = mujoco.mj_name2id(kin.m, mujoco.mjtObj.mjOBJ_BODY, "robot0_link6")
    arm = PiperSDK(a.can, speed_pct=10)
    P, Rs, S = [], [], []
    for i in range(a.n):
        input(f"[{i + 1}/{a.n}] 팔을 새 자세에 두고 Enter ")
        q = arm.joints()
        kin.fk(q)
        P.append(kin.d.xpos[link6] - kin.base)
        Rs.append(kin.d.xmat[link6].reshape(3, 3).copy())
        S.append(arm.end_pose()[:3])
        print(f"   관절(°) {np.round(np.degrees(q), 1)}  SDK 손 끝판(mm) {np.round(S[-1] * 1000, 1)}  모델 link6(mm) {np.round(P[-1] * 1000, 1)}")
    A = np.concatenate(Rs, 0)                       # (3n, 3)
    b = np.concatenate([s - pp for s, pp in zip(S, P)])
    o, *_ = np.linalg.lstsq(A, b, rcond=None)
    res = np.array([np.linalg.norm(s - (pp + R @ o)) for s, pp, R in zip(S, P, Rs)])
    print(f"\nlink6 좌표계에서 고정 차이 o = {np.round(o * 1000, 1)} mm")
    print(f"남는 오차: 평균 {res.mean() * 1000:.1f} mm, 최대 {res.max() * 1000:.1f} mm")
    print("판정:", "같음 (5mm 이하)" if res.max() < 0.005 else "다름 — 관절 0점·부호·DH 오프셋 확인")


if __name__ == "__main__":
    main()
