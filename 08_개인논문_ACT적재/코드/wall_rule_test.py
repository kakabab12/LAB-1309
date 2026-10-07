"""Can the expert execute the v4 fixed-wall grasp over the whole pick range?

Grid: pick yaw -20..+20 deg x the four corners of the +-20 mm pick square, stages 1-3, no jitter.
Run with ACT_DESIGN=v4 (or v2 for the old robot-facing rule).
"""

import json
import math
import os

import numpy as np

import expert as E
from stack_env import PICK_RANGE, SCALE_C, TARGETS, StackEnv, sample_scene

env = StackEnv(render=False)
ex = E.ScriptedExpert(env, np.random.default_rng(0))
home = ex.home_q()
rows = []
for stage in (1, 2, 3):
    for yaw_d in range(-20, 21, 5):
        for cx, cy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            spec = sample_scene(stage, np.random.default_rng(stage * 100 + yaw_d))
            spec.new_bin_xy = SCALE_C + np.array([cx, cy]) * PICK_RANGE
            spec.new_bin_yaw = math.radians(yaw_d)
            env.reset(spec, home)
            bs = env.bin_state(env.active_bin)
            traj = ex.plan(env.qpos(), bs.pos, bs.yaw, TARGETS[stage], jitter=False)
            for a in traj:
                env.step(a)
            env.settle(15)
            r = env.evaluate_stage(stage)
            rows.append({"stage": stage, "yaw": yaw_d, "corner": [cx, cy], "success": bool(r["success"]),
                         "ik_err_mm": 1000 * ex.last_ik_err})
            if not r["success"]:
                print(f"FAIL s{stage} yaw {yaw_d:+d} corner {cx:+d},{cy:+d}: {r}", flush=True)
    ok = [x["success"] for x in rows if x["stage"] == stage]
    print(f"stage {stage}: {sum(ok)}/{len(ok)}", flush=True)
design = os.environ.get("ACT_DESIGN", "v2")
json.dump(rows, open(f"results/wall_rule_{design}.json", "w"), indent=1)
env.close()
