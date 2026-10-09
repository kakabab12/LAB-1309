"""One pallet slot in isolation: earlier slots placed like in the demonstrations (+-3 mm, +-2 deg), the slot's
ACT policy run N times on unseen seeds; prints success and the failure details (xy / z error, tilt, disturbed bins).
usage: python pallet_slot_check.py --slot 6 --run pal_s7 --trials 20 [--noise 0.003]
"""

import argparse
import json
import math

import numpy as np

import gen_pallet as GP
import pallet_grid as PG
import stack_env as SE
from eval_act import Recorder, load_policy, run_stage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", type=int, required=True)  # 0-based
    ap.add_argument("--run", required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--noise", type=float, default=GP.NOISE_XY, help="placement noise of the earlier bins (m)")
    ap.add_argument("--seed0", type=int, default=700000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--te", type=float, default=None, help="ACT temporal ensembling coefficient (re-plan every step)")
    ap.add_argument("--k", type=int, default=None, help="re-plan every k steps instead of the full chunk")
    a = ap.parse_args()
    GP.NOISE_XY = a.noise
    GP.NOISE_YAW = math.radians(2) * a.noise / 0.003
    slots = PG.configure(**GP.LAYOUT)
    env = PG.make_env(slots, render=True)
    home = PG.GridExpert(env).home_q()
    pol = load_policy(f"runs/{a.run}/ckpt_030000", "cpu", a.te, a.k)
    rec = Recorder(env, enabled=False)
    rows = []
    for i in range(a.trials):
        rng = np.random.default_rng(a.seed0 + 1000 * a.slot + i)
        env.reset(GP.scene(slots, a.slot, rng), home)
        refs = {s["name"]: env.bin_state(s["name"]).pos.copy() for s in slots[: a.slot]}
        info = run_stage(env, pol, home, "cpu", rec, horizon=300)
        tgt = PG.slot_target(env, slots, a.slot)
        r = PG.evaluate(env, slots, a.slot, refs, tgt)
        bs = env.bin_state(slots[a.slot]["name"]).pos
        r.update(trial=i, dxy_mm=(1000 * (bs[:2] - tgt[:2])).round(1).tolist(), grasp=info["grasp_site"])
        rows.append(r)
        print(i, "ok" if r["success"] else "FAIL", f"xy {r['xy_err_mm']:.1f} z {r['z_err_mm']:.1f} tilt {r['tilt_deg']:.1f}",
              "d", r["dxy_mm"], r["disturbed"], flush=True)
    k = sum(r["success"] for r in rows)
    print(f"[slot {a.slot}] {k}/{len(rows)}")
    if a.out:
        open(a.out, "w").write(json.dumps({"slot": a.slot, "run": a.run, "noise": a.noise, "te": a.te, "k": a.k, "rows": rows}, indent=1, default=float))


if __name__ == "__main__":
    main()
