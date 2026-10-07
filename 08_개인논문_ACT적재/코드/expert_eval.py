"""The scripted expert on the per-stage evaluation scenes (same seeds as eval_act.py), with the same
RETURN_HOME + settle as the policy evaluation. Reports success and the final tilt of the placed bin.
usage: ACT_DESIGN=v5 python expert_eval.py --stages 1 2 3 --n 50 [--jitter]
"""

import argparse
import json
import os

import numpy as np

from eval_act import EVAL_SEED0
from expert import ScriptedExpert
from stack_env import BIN_H, BIN_NAMES, TARGETS, StackEnv, sample_scene

ap = argparse.ArgumentParser()
ap.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3])
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--jitter", action="store_true", help="speed/waypoint jitter as in the demonstrations")
a = ap.parse_args()

env = StackEnv(render=False)
ex = ScriptedExpert(env, np.random.default_rng(0))
home = ex.home_q()
out = {}
for stage in a.stages:
    rows = []
    for i in range(a.n):
        env.reset(sample_scene(stage, np.random.default_rng(EVAL_SEED0 + 1000 * stage + i)), home)
        refs = {n: env.bin_state(n).pos.copy() for n in BIN_NAMES[: stage - 1]}
        bs = env.bin_state(BIN_NAMES[stage - 1])
        tgt = TARGETS[stage] if stage < 3 else env.bin_state("bin_a").pos + np.array([0, 0, BIN_H])
        for act in ex.plan(env.qpos(), bs.pos, bs.yaw, tgt, jitter=a.jitter):
            env.step(act)
        q0 = env.qpos()
        for k in range(30):
            env.step(q0 + (home - q0) * (k + 1) / 30)
        env.settle(15)
        r = env.evaluate_stage(stage, ref_positions=refs)
        rows.append({"trial": i, "success": bool(r["success"]), "tilt": float(r["tilt_deg"]), "xy": float(r["xy_err_mm"])})
    t = np.array([x["tilt"] for x in rows])
    out[stage] = {"success": sum(x["success"] for x in rows), "n": len(rows), "tilt_mean": float(t.mean()),
                  "tilt_max": float(t.max()), "tilt_gt3": int((t > 3).sum()), "tilt_gt5": int((t > 5).sum())}
    print(f"stage {stage}: {out[stage]}", flush=True)
design = os.environ.get("ACT_DESIGN", "v2")
json.dump(out, open(f"results/expert_eval_{design}{'_jitter' if a.jitter else ''}.json", "w"), indent=1)
env.close()
