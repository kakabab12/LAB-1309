"""How much grasp-position error does each demonstration design tolerate?

Shifts the expert's grasp point along the wall normal (negative = toward/onto the wall) and checks
whether stage 1 still succeeds. Designs: clearance 1.5 mm / open 0.32 rad (v1, current data) vs
clearance 10 mm / open 0.55 rad (v2, robust).
"""

import json
import math

import numpy as np

import expert as E
from stack_env import BIN_NAMES, TARGETS, StackEnv, sample_scene

OFFSETS = [-12, -9, -6, -3, 0, 3, 6, 9]  # mm along the facing-wall normal
import sys
DESIGNS = {"v2a (여유 7mm, 열림 0.32)": (0.007, 0.32), "v2b (여유 9mm, 열림 0.32)": (0.009, 0.32)}
N = 4

env = StackEnv(render=False)
res = {}
for dname, (clear, opening) in DESIGNS.items():
    ex = E.ScriptedExpert(env, np.random.default_rng(3))
    ex.CLEAR = clear
    E.GRIPPER_OPEN = opening  # plan() reads the module constant
    home = ex.home_q()
    row = {}
    for off in OFFSETS:
        rng = np.random.default_rng(100)
        ok = 0
        for _ in range(N):
            env.reset(sample_scene(1, rng), home)
            bs = env.bin_state(BIN_NAMES[0])
            n = ex.facing_wall_normal(bs.pos[:2], bs.yaw)
            ex_clear = ex.CLEAR
            ex.CLEAR = clear + off / 1000.0  # move the whole grasp approach along the normal
            traj = ex.plan(env.qpos(), bs.pos, bs.yaw, TARGETS[1], jitter=False)
            ex.CLEAR = ex_clear
            for a in traj:
                env.step(a)
            env.settle(15)
            ok += env.evaluate_stage(1)["success"]
        row[off] = ok / N
        print(f"{dname} offset {off:+d}mm: {ok}/{N}", flush=True)
    res[dname] = row
json.dump(res, open("results/grasp_tolerance_v2.json", "w"), indent=1, ensure_ascii=False)
print("saved results/grasp_tolerance.json")
env.close()
