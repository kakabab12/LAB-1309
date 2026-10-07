"""Expert on the widened DR pick range (+-28 mm, +-30 deg): extreme yaws x extreme corners, with jitter.
usage: ACT_DESIGN=v4 python wide_expert_test.py 1 2 ;  ACT_DESIGN=v5 python wide_expert_test.py 3"""
import math, sys, json, os
import numpy as np
from expert import ScriptedExpert
from stack_env import BIN_H, BIN_NAMES, PICK_RANGE_DR, SCALE_C, TARGETS, StackEnv, sample_scene
env = StackEnv(render=False, dr=True)
ex = ScriptedExpert(env, np.random.default_rng(1))
home = ex.home_q()
out = {}
for stage in map(int, sys.argv[1:]):
    ok = n = 0
    for yaw_d in (-30, -25, 25, 30):
        for cx, cy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            spec = sample_scene(stage, np.random.default_rng(stage * 1000 + yaw_d + 7 * cx + 3 * cy))
            spec.new_bin_xy = SCALE_C + np.array([cx, cy]) * PICK_RANGE_DR
            spec.new_bin_yaw = math.radians(yaw_d)
            env.reset(spec, home)
            refs = {k: env.bin_state(k).pos.copy() for k in BIN_NAMES[: stage - 1]}
            bs = env.bin_state(BIN_NAMES[stage - 1])
            tgt = TARGETS[stage] if stage < 3 else env.bin_state("bin_a").pos + np.array([0, 0, BIN_H])
            for a in ex.plan(env.qpos(), bs.pos, bs.yaw, tgt):
                env.step(a)
            env.settle(15)
            r = env.evaluate_stage(stage, ref_positions=refs)
            ok += r["success"]; n += 1
            if not r["success"]:
                print(f"FAIL s{stage} yaw {yaw_d} corner {cx},{cy}: xy {r['xy_err_mm']:.1f} tilt {r['tilt_deg']:.1f}", flush=True)
    out[stage] = [ok, n]
    print(f"stage {stage}: {ok}/{n}", flush=True)
json.dump(out, open(f"results/wide_expert_{os.environ.get('ACT_DESIGN','v2')}.json", "w"))
