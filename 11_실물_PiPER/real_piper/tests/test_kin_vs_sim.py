"""kin.py 가 시뮬레이터와 같은 값을 내는지 (CPU, GPU 안 씀):
   ① 관절 → 손끝 위치·방향·손 사원수 가 시뮬 관측(robot0_eef_pos/quat, 제어기 ee_ori_mat)과 같은가
   ② 같은 동작에서 kin.goal + kin.ik 의 관절 목표가 시뮬 제어기(PiperIKController.q_goal)와 같은가
   CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PIPER_NOPED=1 PIPER_WORLD=piper_sim/worlds/default.json \
       python real_piper/tests/test_kin_vs_sim.py"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import piper_sim.piper_robot as pr  # noqa: E402

pr.use_piper_in_lerobot()
import piper_sim.world as W  # noqa: E402

W.install()
import piper_sim.world_expert as WE  # noqa: E402
from real_piper.kin import Kin  # noqa: E402

sc = W.sample_scene(1000, n_objects=3, n_pads=2)
ep = WE.make_episode(sc, 1000)
ctrl = ep.inner.robots[0].controller
kin = Kin()
rng = np.random.default_rng(0)
e_fk = e_q = e_mat = 0.0
for t in range(60):
    o = ep.obs["robot_state"]
    q = np.asarray(o["joints"]["pos"], dtype=np.float64)
    p, M, hq = kin.fk(q)
    e_fk = max(e_fk, np.abs(p - o["eef"]["pos"]).max(), min(np.abs(hq - o["eef"]["quat"]).max(), np.abs(hq + o["eef"]["quat"]).max()))
    e_mat = max(e_mat, np.abs(M - np.asarray(o["eef"]["mat"])).max())
    a = np.r_[rng.uniform(-0.6, 0.6, 3), rng.uniform(-0.15, 0.15, 3) * (t % 5 != 0), [-1.0]].astype(np.float32)
    kin.goal_mat = ctrl.goal_ori.copy() if getattr(ctrl, "goal_ori", None) is not None else None
    gp, gM = kin.goal(q, a)
    q_mine = kin.ik(gp, gM, q)
    ep.step(a, "T0")                      # 시뮬 제어기가 같은 동작으로 set_goal → q_goal
    q_sim = np.asarray(ctrl.q_goal)
    e_q = max(e_q, np.abs(q_mine - q_sim).max())
print(f"① 손끝 위치·손 사원수 최대 차이 {e_fk:.2e}, 손끝 방향행렬 {e_mat:.2e}")
print(f"② 관절 목표 최대 차이 {np.degrees(e_q):.4f}° (같은 동작)")
ep.env.close()
