"""실물용 코드 길 전체(관절 → kin → VLA → 변화량 → kin 역기구학 → 관절 목표)를 시뮬 팔로 여러 장면 돌려,
   시뮬 평가 길(Episode.run_policy: 시뮬 제어기가 변화량을 받음)과 성공 수를 견준다. 둘 다 동작 변수(속도·잡음·물리) 없이, 지연은 장면 값.
   VLA 혼자 (다듬기 없음). PIPER_NOPED=1 MUJOCO_GL=egl python real_piper/tests/test_sim_success.py --seeds 1000-1011"""
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
p.add_argument("--seeds", default="1000-1011")
p.add_argument("--model", default="outputs/piper_w12m_model/merged")
p.add_argument("--steps", type=int, default=600)
a = p.parse_args()
lo, hi = map(int, a.seeds.split("-"))
vla = VLA(ROOT / a.model, latency_steps=9, cuda_graph=True)
W = vla.W
kin = Kin()
res = []
for seed in range(lo, hi + 1):
    mv = W.sample_motion(seed, 1)
    L = int(mv["latency"]) if mv and mv.get("latency") is not None else 9
    # ① 실물용 길
    robot = robot_io.SimRobot(seed, vla.r)
    instr = robot.instruction
    planner = Planner(vla, latency_steps=L, mode="steps")
    kin.reset()
    ok1, close = False, False
    for t in range(a.steps):
        q = robot.joints()
        px = robot.obs()["pixels"]
        obs = make_obs(px["image"], px["image2"], kin, q, robot.gripper(), 0.0)
        eef = obs["robot_state"]["eef"]["pos"]
        act = planner.next(obs, instr, eef)
        d = to_delta(act, eef, vla.abs_pos)
        gp, gM = kin.goal(q, d)
        robot.send_joints(kin.ik(gp, gM, q))
        close = True if d[6] > 0.1 else (False if d[6] < -0.1 else close)
        robot.send_gripper(close)
        robot.tick()
        if robot.success():
            ok1 = True
            break
    t1 = t + 1
    robot.close()
    # ② 시뮬 평가 길 (같은 장면, 같은 지연)
    robot = robot_io.SimRobot(seed, vla.r)
    robot.ctrl.set_goal = type(robot.ctrl).set_goal.__get__(robot.ctrl)     # 관절 바로 넣기를 끔 → 시뮬 제어기 그대로
    vla.r.args.latency_steps = L
    why, n = robot.ep.run_policy(instr, "T0", a.steps, success_fn=robot.goal_fn)
    ok2 = why == "success"
    robot.close()
    res.append((seed, L, ok1, t1, ok2, n))
    print(f"seed {seed} 지연 {L}: 실물용 길 {'성공' if ok1 else '실패'} ({t1}) | 시뮬 평가 길 {'성공' if ok2 else '실패'} ({n})  {instr}", flush=True)
r = np.array([(x[2], x[4]) for x in res])
print(f"SUMMARY 실물용 길 {int(r[:, 0].sum())}/{len(r)}, 시뮬 평가 길 {int(r[:, 1].sum())}/{len(r)}")
