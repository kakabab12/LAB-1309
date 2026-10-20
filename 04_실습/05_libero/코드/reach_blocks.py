#!/usr/bin/env python
"""블록 장면에서 받침대 높이·밑동 위치별로 손이 닿는지 (10/20, 사용자 "받침대 없이로 해주지").
   그리퍼가 아래를 보고(기울기 0·±15°, 손목 방향 0·90°) 블록 쥐는 점·8cm 위·판에 놓는 점·쌓는 점에 닿는지 역기구학으로.
   넓은 배치(블록 ±4.5cm, 판 ±3.5cm)의 네 모서리까지. python piper_sim/reach_blocks.py"""
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
m = mujoco.MjModel.from_xml_path(str(HERE.parent / "third_party/mujoco_menagerie/agilex_piper/piper.xml"))
d = mujoco.MjData(m)
lo = np.radians([-150, 0, -154.5, -100, -70, -120]); hi = np.radians([150, 180, 0, 100, 70, 120])
b6 = m.body("link6").id
RZ90 = R.from_euler("z", 90, degrees=True).as_matrix()
DOWN = np.array([[0.001, 0.9985, -0.0555], [1.0, -0.0009, 0.0006], [0.0005, -0.0555, -0.9985]])   # 처음 자세 방향
TABLE = 0.90
REG = {"red": (-0.09, 0.00, 0.045), "green": (-0.09, 0.13, 0.045), "blue": (-0.09, -0.13, 0.045), "yellow": (-0.10, -0.26, 0.045),
       "purple": (0.04, 0.00, 0.035), "gray": (0.04, -0.16, 0.035), "orange": (-0.13, 0.27, 0.035), "white": (0.04, 0.16, 0.035)}


def fk(q):
    d.qpos[:6] = q
    mujoco.mj_kinematics(m, d)
    R6 = d.xmat[b6].reshape(3, 3)
    return d.xpos[b6] + R6 @ [0, 0, 0.12], R6 @ RZ90


def ik(target, Rdes, q0, iters=300):
    q = q0.copy()
    for _ in range(iters):
        p, E = fk(q)
        e = np.r_[target - p, (R.from_matrix(Rdes) * R.from_matrix(E).inv()).as_rotvec()]
        if np.linalg.norm(e[:3]) < 1e-4 and np.linalg.norm(e[3:]) < 1e-3:
            break
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        d.qpos[:6] = q; mujoco.mj_forward(m, d)
        mujoco.mj_jac(m, d, jp, jr, p, b6)
        J = np.vstack([jp[:, :6], jr[:, :6]])
        q = np.clip(q + 0.5 * J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e), lo, hi)
    p, E = fk(q)
    return q, np.linalg.norm(target - p), np.degrees((R.from_matrix(Rdes) * R.from_matrix(E).inv()).magnitude())


def reachable(pt, base, rng):
    """여러 손 방향 중 하나라도 닿으면 (위치 1cm·방향 5° 안) 관절 여유(도)를, 아니면 None."""
    rad = (pt - base)[:2]; rad = rad / (np.linalg.norm(rad) + 1e-9)
    axis = np.array([-rad[1], rad[0], 0.0])
    best = None
    for yaw in (0, 90):
        for tilt in (0, 15, -15):
            Rd = R.from_rotvec(axis * np.radians(tilt)).as_matrix() @ R.from_euler("z", yaw, degrees=True).as_matrix() @ DOWN
            for s in range(4):
                q, pe, oe = ik(pt - base, Rd, rng.uniform(lo, hi))
                if pe < 0.01 and oe < 5:
                    mg = float(np.degrees(np.min(np.minimum(q - lo, hi - q))))
                    best = mg if best is None else max(best, mg)
                    break
    return best


def main():
    rng = np.random.default_rng(0)
    for bx, ped in ((-0.38, 0.29), (-0.38, 0.0), (-0.33, 0.0), (-0.30, 0.0)):
        base = np.array([bx, 0.0, TABLE + ped])
        miss, margins = [], []
        for name, (x, y, jit) in REG.items():
            block = name in ("red", "green", "blue", "yellow")
            for cx in (-jit, jit):
                for cy in (-jit, jit):
                    pts = [("쥐기", TABLE + 0.0225), ("8cm 위", TABLE + 0.1025)] if block else \
                          [("놓기", TABLE + 0.005 + 0.0225 + 0.015), ("쌓기 높이", TABLE + 0.045 + 0.0225 + 0.015 + 0.04)]
                    for lab, z in pts:
                        r = reachable(np.array([x + cx, y + cy, z]), base, rng)
                        if r is None:
                            miss.append(f"{name}({x + cx:+.2f},{y + cy:+.2f}) {lab}")
                        else:
                            margins.append(r)
        print(f"밑동 x {bx:+.2f}, 받침대 {ped * 100:.0f}cm: 닿지 않음 {len(miss)} / {len(miss) + len(margins)}, "
              f"관절 여유 최소 {min(margins):.0f}° 중앙 {np.median(margins):.0f}°", flush=True)
        for s in miss[:8]:
            print("   ", s)


if __name__ == "__main__":
    main()
