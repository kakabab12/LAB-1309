"""Feasibility of 2x2 x 2-layer palletising with the scripted expert (8 bins in a row).

usage: python pallet_test.py [sequences] [--gif]
"""

import json
import sys

import mujoco
import numpy as np
from PIL import Image

from pallet_env import BIN_NAMES, TARGETS, BIN_H, PalletEnv, below, sample_pick, sample_scene
from pallet_expert import PalletExpert

N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3
GIF = "--gif" in sys.argv
env = PalletEnv(render=False)
ex = PalletExpert(env, np.random.default_rng(11))
home = ex.home_q()
rng = np.random.default_rng(2024)
big = mujoco.Renderer(env.m, 360, 480)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = [0.16, 0.02, 0.05]
cam.distance = 0.6
cam.azimuth = 210
cam.elevation = -32
frames, res = [], []
for seq in range(N):
    env.reset(sample_scene(1, rng), home)
    row = []
    for stage in range(1, 9):
        if stage > 1:
            xy, yaw = sample_pick(rng)
            env.spawn_next_bin(stage, xy, yaw)
        refs = {n: env.bin_state(n).pos.copy() for n in BIN_NAMES[: stage - 1]}
        bs = env.bin_state(BIN_NAMES[stage - 1])
        tgt = TARGETS[stage] if not below(stage) else env.bin_state(BIN_NAMES[below(stage) - 1]).pos + np.array([0, 0, BIN_H])
        for t, a in enumerate(ex.plan(env.qpos(), bs.pos, bs.yaw, tgt)):
            env.step(a)
            if GIF and seq == 0 and t % 6 == 0:
                big.update_scene(env.d, cam)
                frames.append(Image.fromarray(big.render()))
        q0 = env.qpos()
        for i in range(30):
            env.step(q0 + (home - q0) * (i + 1) / 30)
        env.settle(15)
        r = env.evaluate_stage(stage, ref_positions=refs)
        row.append({"stage": stage, "success": r["success"], "xy_err_mm": round(r["xy_err_mm"], 1),
                    "disturbed": r["disturbed"], "ik_err_mm": round(ex.last_ik_err * 1000, 1)})
    res.append(row)
    print(f"seq {seq}: " + " ".join(f"{x['stage']}:{'o' if x['success'] else 'x'}({x['xy_err_mm']})" for x in row), flush=True)
    if seq == 0:
        big.update_scene(env.d, cam)
        Image.fromarray(big.render()).save("results/pallet_expert_final.png")
ok = np.array([[x["success"] for x in r] for r in res])
print("per-stage success:", ok.mean(0).round(2).tolist(), "| full 8-stack:", ok.all(1).mean())
print("max IK err mm:", max(x["ik_err_mm"] for r in res for x in r))
json.dump(res, open("results/pallet_expert_test.json", "w"), indent=1)
if frames:
    frames[0].save("results/gifs/pallet_expert_seq0.gif", save_all=True, append_images=frames[1:], duration=120, loop=0, optimize=True)
big.close()
env.close()
