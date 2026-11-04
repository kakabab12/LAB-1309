"""두 길(실물용 / 시뮬 평가)을 추론 잡음까지 같게 (추론마다 같은 seed) 돌려 손 궤적이 언제 갈라지는지 (GPU).
   PIPER_NOPED=1 MUJOCO_GL=egl python real_piper/tests/test_traj_same_noise.py --seed 1003"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from real_piper.kin import Kin  # noqa: E402
from real_piper.policy import VLA, Planner, make_obs, to_delta  # noqa: E402
from real_piper import robot_io  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=1003)
p.add_argument("--model", default="outputs/piper_w12m_model/merged")
p.add_argument("--steps", type=int, default=600)
p.add_argument("--exact-state", action="store_true", help="실물용 길에도 시뮬 관측의 손끝·손가락 값을 그대로 (차이 원인 가르기)")
a = p.parse_args()
vla = VLA(ROOT / a.model, latency_steps=9, cuda_graph=True)
torch, W = vla.torch, vla.W
cnt = {"n": 0}
orig = vla.r.policy.predict_action_chunk


def seeded(*args, **kw):
    torch.manual_seed(1000 + cnt["n"])
    cnt["n"] += 1
    return orig(*args, **kw)
vla.r.policy.predict_action_chunk = seeded
mv = W.sample_motion(a.seed, 1)
L = int(mv["latency"]) if mv and mv.get("latency") is not None else 9
kin = Kin()
# ① 실물용 길
robot = robot_io.SimRobot(a.seed, vla.r)
instr = robot.instruction
planner = Planner(vla, latency_steps=L, mode="steps")
H1, close = [], False
cnt["n"] = 0
for t in range(a.steps):
    q = robot.joints()
    px = robot.obs()["pixels"]
    obs = make_obs(px["image"], px["image2"], kin, q, robot.gripper(), 0.0)
    if a.exact_state:
        rs = robot.obs()["robot_state"]
        obs["robot_state"]["eef"]["pos"] = np.asarray(rs["eef"]["pos"]).copy()
        obs["robot_state"]["gripper"]["qpos"] = np.asarray(rs["gripper"]["qpos"]).copy()
    eef = obs["robot_state"]["eef"]["pos"]
    act = planner.next(obs, instr, eef)
    d = to_delta(act, eef, vla.abs_pos)
    gp, gM = kin.goal(q, d)
    robot.send_joints(kin.ik(gp, gM, q))
    close = True if d[6] > 0.1 else (False if d[6] < -0.1 else close)
    robot.send_gripper(close)
    robot.tick()
    H1.append(np.asarray(robot.obs()["robot_state"]["eef"]["pos"]).copy())
    if robot.success():
        break
ok1 = robot.success()
robot.close()
# ② 시뮬 평가 길
robot = robot_io.SimRobot(a.seed, vla.r)
robot.ctrl.set_goal = type(robot.ctrl).set_goal.__get__(robot.ctrl)
vla.r.args.latency_steps = L
H2 = []
orig_step = robot.ep.step.__func__


def step(self, action, phase):
    orig_step(self, action, phase)
    H2.append(np.asarray(self.obs["robot_state"]["eef"]["pos"]).copy())
robot.ep.step = step.__get__(robot.ep)
cnt["n"] = 0
why, n = robot.ep.run_policy(instr, "T0", a.steps, success_fn=robot.goal_fn)
robot.close()
m = min(len(H1), len(H2))
dd = np.linalg.norm(np.array(H1[:m]) - np.array(H2[:m]), axis=1) * 100
first = next((i for i, x in enumerate(dd) if x > 1.0), None)
print(f"seed {a.seed} 지연 {L} {'(손끝·손가락 시뮬 값)' if a.exact_state else ''}: 실물용 {'성공' if ok1 else '실패'} {len(H1)}스텝, 시뮬 평가 {why} {n}스텝")
print(f"  손끝 차이 (cm): 50스텝 {dd[min(49, m-1)]:.2f}, 100 {dd[min(99, m-1)]:.2f}, 200 {dd[min(199, m-1)]:.2f}, 최대 {dd.max():.2f}; 1cm 넘는 첫 스텝 {first}")
