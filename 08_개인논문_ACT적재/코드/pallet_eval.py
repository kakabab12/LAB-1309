"""Evaluate the 2x2 x 2-layer pallet: 8 slot policies run one after another (like the TCP-triggered cell),
RETURN_HOME between slots, optional scale-verified retry. Policy on the CPU (one chunk per 3.3 s).
Scenes: pick poses from seeds 800000+i (disjoint from the demonstration seeds).
usage: ACT_DESIGN=v4 python pallet_eval.py --runs pal_s1 ... pal_s8 --start 0 --trials 10 --out results/eval/pallet_p0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import pallet_grid as PG
import stack_env as SE
from eval_act import Recorder, load_policy, run_stage
from eval_retry import on_scale
from gen_pallet import LAYOUT

SEED0 = 800000
HORIZON = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="one run per slot in placement order; 4 runs = the first layer only (2x2)")
    ap.add_argument("--step", type=int, default=30000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--gifs", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=None, help="re-plan every k steps (default: run each 100-step chunk fully)")
    a = ap.parse_args()
    torch.set_num_threads(2)
    slots = PG.configure(**LAYOUT)
    env = PG.make_env(slots, render=True)
    home = PG.GridExpert(env).home_q()
    pols = [load_policy(f"runs/{r}/ckpt_{a.step:06d}", "cpu", None, a.k) for r in a.runs]
    rec = Recorder(env, enabled=a.gifs > 0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seqs = []
    for i in range(a.start, a.start + a.trials):
        rng = np.random.default_rng(SEED0 + i)
        xy, yaw = SE.sample_pick(rng)
        env.reset(SE.SceneSpec(stage=1, new_bin_xy=xy, new_bin_yaw=yaw), home)
        rec.enabled = i - a.start < a.gifs
        rows = []
        for k in range(len(pols)):
            if k > 0:
                xy, yaw = SE.sample_pick(rng)
                env.spawn_next_bin(k + 1, xy, yaw)
            refs = {s["name"]: env.bin_state(s["name"]).pos.copy() for s in slots[:k]}
            attempts, first = 0, None
            while True:
                attempts += 1
                run_stage(env, pols[k], home, "cpu", rec, horizon=HORIZON)
                tgt = PG.slot_target(env, slots, k)
                r = PG.evaluate(env, slots, k, refs, tgt)
                first = r["success"] if first is None else first
                if attempts >= a.max_attempts or not on_scale(env, slots[k]["name"]):
                    break
            r.update(slot=k, attempts=attempts, first_attempt_success=bool(first),
                     bin_pos=env.bin_state(slots[k]["name"]).pos.tolist(), target=tgt.tolist())
            rows.append(r)
        ok = [x["success"] for x in rows]
        ok0 = [x["first_attempt_success"] for x in rows]
        seqs.append({"trial": i, "slots": rows, "cumulative": [all(ok[: k + 1]) for k in range(len(pols))],
                     "cumulative_first": [all(ok0[: k + 1]) for k in range(len(pols))]})
        rec.save(out / "gifs" / f"pallet_t{i:02d}_{''.join('o' if o else 'x' for o in ok)}.gif")
        print(f"seq {i}: {''.join('o' if o else 'x' for o in ok)} attempts {[x['attempts'] for x in rows]}", flush=True)
    res = {"runs": a.runs, "k": a.k, "n": len(seqs), "slot_success": np.mean([[x["success"] for x in s["slots"]] for s in seqs], 0).tolist(),
           "cumulative_success": np.mean([s["cumulative"] for s in seqs], 0).tolist(),
           "cumulative_first_attempt": np.mean([s["cumulative_first"] for s in seqs], 0).tolist(), "sequences": seqs}
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"[pallet] full {len(pols)}-stack", res["cumulative_success"][-1], "first attempt", res["cumulative_first_attempt"][-1], flush=True)
    rec.close()
    env.close()


if __name__ == "__main__":
    main()
