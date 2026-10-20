#!/usr/bin/env python
"""PiPER 시작 관절 각도: 손끝을 Panda 와 같은 처음 자세(위치·방향)에 두는 역기구학 (2026-10-03).
   python piper_sim/calib_home.py  → piper_sim/init_qpos.json"""
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).resolve().parent
m = mujoco.MjModel.from_xml_path(str(HERE.parent / "third_party/mujoco_menagerie/agilex_piper/piper.xml"))
d = mujoco.MjData(m)
lo = np.radians([-150, 0, -154.5, -100, -70, -120]); hi = np.radians([150, 180, 0, 100, 70, 120])
b6 = m.body("link6").id
import os as _os
_SFX = "_noped" if _os.environ.get("PIPER_NOPED") == "1" else ""     # 10/20: 받침대 없이
BASE = np.array(json.loads((HERE / f"layout{_SFX}.json").read_text())["base"])
RZ90 = R.from_euler("z", 90, degrees=True).as_matrix()
HOME_POS = np.array([-0.2071, 0.0019, 1.1786])
HOME_MAT = np.array([[0.001, 0.9985, -0.0555], [1.0, -0.0009, 0.0006], [0.0005, -0.0555, -0.9985]])


def fk(q):
    d.qpos[:6] = q
    mujoco.mj_kinematics(m, d)
    R6 = d.xmat[b6].reshape(3, 3)
    return d.xpos[b6] + R6 @ [0, 0, 0.12], R6 @ RZ90          # 손끝 기준점, 그 프레임 E


def ik(target, Rdes, q0, iters=400):
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


# Panda 처음 자세 그대로는 J5 가 한계(70°)에 걸린다 → 근처(±6cm, 위아래 −8~+2cm)에서
# 방향은 같게(그리퍼 아래, 손가락 y 방향) 두고 관절 여유가 가장 큰 자리를 찾는다
rng = np.random.default_rng(0)
best = None
# 10/20 받침대 없음: 밑동이 낮아 Panda 처음 자세 근처(책상 위 28cm, 밑동 앞 12cm)는 팔을 접어도 닿지 않는다 → 앞으로·아래로 넓게
#   손목 5번 관절(±70°) 때문에 그리퍼를 똑바로 아래로 둘 수 없어(책상 위 10cm 넘으면) 앞으로 기울인 방향도 찾는다 (TILTS)
GRID = ((np.arange(0.0, 0.121, 0.04), np.array([0.0]), np.arange(-0.14, -0.019, 0.04)) if _SFX
        else (np.arange(-0.06, 0.061, 0.03), np.arange(-0.06, 0.061, 0.03), np.arange(-0.08, 0.021, 0.02)))
TILTS = (-30, -20, -10, 0, 10, 20, 30) if _SFX else (0,)
for dx in GRID[0]:
    for dy in GRID[1]:
        for dz in GRID[2]:
          tgt = HOME_POS + [dx, dy, dz]
          for tilt in TILTS:
            Rh = R.from_euler("y", tilt, degrees=True).as_matrix() @ HOME_MAT
            for s in range(8):
                q, pe, oe = ik(tgt - BASE, Rh, rng.uniform(lo, hi))
                if pe > 1e-3 or oe > 0.5:
                    continue
                margin = np.min(np.minimum(q - lo, hi - q))
                if best is None or margin > best[4]:
                    best = (tgt, q, pe, oe, margin, tilt, Rh)
tgt, q, pe, oe, margin, tilt, Rh = best
print("기울기", tilt, "°")
print("처음 자세 손끝 위치", np.round(tgt, 3).tolist(), "(Panda", HOME_POS.tolist(), ")")
print("init_qpos (rad)", np.round(q, 4).tolist(), "deg", np.round(np.degrees(q), 1).tolist())
print(f"위치 오차 {pe * 1000:.2f} mm, 방향 오차 {oe:.2f}°, 관절 한계 여유 {np.degrees(margin):.1f}°")
(HERE / f"init_qpos{_SFX}.json").write_text(json.dumps({"qpos": q.tolist(), "home_pos": tgt.tolist(), "pos_err_mm": pe * 1000,
                                                 "ori_err_deg": oe, "joint_margin_deg": float(np.degrees(margin)),
                                                 "tilt_deg": float(tilt), "home_mat": Rh.tolist()}))
