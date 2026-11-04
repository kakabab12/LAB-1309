"""실물용 관측 만들기(make_obs: 관절 → kin 정기구학)가 시뮬 관측과 같은 VLA 입력·출력을 내는지 (GPU).
   같은 장면·같은 순간에서 ① 시뮬이 준 관측 ② 관절·그리퍼만으로 다시 만든 관측을 같은 잡음 seed 로 VLA 에 넣고
   상태 입력 8+24 칸과 첫 동작 묶음을 견준다. 여러 순간 (시뮬 평가 경로로 손을 움직이며).
   PIPER_NOPED=1 MUJOCO_GL=egl python real_piper/tests/test_obs_equiv.py --seeds 1000,1003 --model outputs/piper_w12m_model/merged"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from real_piper.kin import Kin  # noqa: E402
from real_piper.policy import VLA, make_obs  # noqa: E402
from real_piper import robot_io  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--seeds", default="1000,1003")
p.add_argument("--model", default="outputs/piper_w12m_model/merged")
a = p.parse_args()
vla = VLA(ROOT / a.model, latency_steps=9, cuda_graph=True)
torch = vla.torch
kin = Kin()
captured = {}
orig_prep = vla.r.prep


def prep(batch, obs):
    b = orig_prep(batch, obs)
    captured["state"] = b["observation.state"].detach().float().cpu().numpy()[0]
    return b
vla.r.prep = prep
worst_s = worst_a = 0.0
for seed in map(int, a.seeds.split(",")):
    robot = robot_io.SimRobot(seed, vla.r)
    instr = robot.instruction
    for k in range(6):
        o_sim = robot.obs()
        o_me = make_obs(o_sim["pixels"]["image"], o_sim["pixels"]["image2"], kin, robot.joints(), robot.gripper(), 0.0)
        vla.reset(); torch.manual_seed(7); a1, _ = vla.infer(o_sim, instr); s1 = captured["state"].copy()
        vla.reset(); torch.manual_seed(7); a2, _ = vla.infer(o_me, instr); s2 = captured["state"].copy()
        ds, da = np.abs(s1 - s2).max(), np.abs(a1 - a2).max()
        worst_s, worst_a = max(worst_s, ds), max(worst_a, da)
        print(f"seed {seed} 순간 {k}: 상태 입력 최대 차이 {ds:.4f} (정규화 공간), 동작 묶음 최대 차이 {da:.4f} (위치 m / ±1)", flush=True)
        robot._q_cmd = None                           # 관절 바로 넣기 끄고 시뮬 제어기로
        for i in range(15):                           # VLA 첫 동작들로 손을 조금 움직여 다른 순간을 만든다
            j = min(i, len(a1) - 1)
            eef = robot.obs()["robot_state"]["eef"]["pos"]
            robot.ep.step(np.r_[np.clip((a1[j, :3] - eef) / 0.05, -1, 1), a1[j, 3:6], a1[j, 6]].astype(np.float32), "T0")
    robot.close()
print(f"전체 최대: 상태 입력 {worst_s:.4f}, 동작 {worst_a:.4f}")
