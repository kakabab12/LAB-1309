"""Chained evaluation with scale-verified retry (system-level recovery using the cell's own weight sensor).

In the real cell the Jetson reads the scale after every RETURN_HOME. If the bin is still on the scale
(reading >= 118 g) the pick failed, so the same stage is triggered again over the TCP trigger, up to --max-attempts.
The scale cannot see a bad placement, so only failed picks are retried; the success test is unchanged.
Same chained scenes (seeds) as eval_act.py, so the results compare directly with its 'chained' numbers.
usage: python eval_retry.py --ckpt S1 S2 S3 --out results/eval/v5_n100_retry --trials 50 --max-attempts 3
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from eval_act import EVAL_SEED0, Recorder, load_policy, run_stage
from expert import ScriptedExpert
from stack_env import BIN_H, BIN_NAMES, SCALE_C, SCALE_H, SCALE_HALF, StackEnv, sample_pick, sample_scene


G = 9.81
SCALE_TRIGGER_G = 60.0  # half of a full bin (123 g): above it the bin is still on the scale


def scale_reading_g(env: StackEnv) -> float:
    """Simulated scale display: total normal contact force on the scale body, in grams (what the 7-segment shows)."""
    m, d = env.m, env.d
    sb = m.body("scale").id
    f6 = np.zeros(6)
    total = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if sb in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]):
            mujoco.mj_contactForce(m, d, i, f6)
            total += abs(f6[0])
    return 1000.0 * total / G


def on_scale(env: StackEnv, name: str) -> bool:
    """What the weight reading would say: the bin still stands on the scale plate."""
    bs = env.bin_state(name)
    inside = np.all(np.abs(bs.pos[:2] - SCALE_C) < SCALE_HALF - 0.005)
    return bool(inside and abs(bs.pos[2] - (SCALE_H + BIN_H / 2)) < 0.015)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs=3, required=True)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--start", type=int, default=0, help="first sequence index (to split a long evaluation)")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gifs", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trigger", choices=["position", "weight"], default="position",
                    help="retry when the bin is still on the scale: by its pose, or by the simulated scale reading")
    a = ap.parse_args()
    torch.set_num_threads(2)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = StackEnv(render=True)
    home = ScriptedExpert(env).home_q()
    pols = [load_policy(p, a.device, None) for p in a.ckpt]
    rec = Recorder(env, enabled=a.gifs > 0)
    seqs = []
    for i in range(a.start, a.start + a.trials):
        rng = np.random.default_rng(EVAL_SEED0 + 90000 + i)
        env.reset(sample_scene(1, rng), home)
        rec.enabled = i - a.start < a.gifs
        seq = []
        for stage in (1, 2, 3):
            if stage > 1:
                xy, yaw = sample_pick(rng)
                env.spawn_next_bin(stage, xy, yaw)
            refs = {n: env.bin_state(n).pos.copy() for n in BIN_NAMES[: stage - 1]}
            name = BIN_NAMES[stage - 1]
            attempts, first = 0, None
            readings = []
            while True:
                attempts += 1
                run_stage(env, pols[stage - 1], home, a.device, rec)
                r = env.evaluate_stage(stage, ref_positions=refs)
                if first is None:
                    first = bool(r["success"])
                grams = scale_reading_g(env)
                still = grams >= SCALE_TRIGGER_G if a.trigger == "weight" else on_scale(env, name)
                readings.append({"attempt": attempts, "scale_g": round(grams, 1), "on_scale_pose": on_scale(env, name)})
                if attempts >= a.max_attempts or not still:
                    break
            r.update(attempts=attempts, first_attempt_success=first, scale=readings)
            seq.append(r)
        ok = [s["success"] for s in seq]
        first_ok = [s["first_attempt_success"] for s in seq]
        seqs.append({"trial": i, "stages": seq, "cumulative": [all(ok[: k + 1]) for k in range(3)],
                     "cumulative_first": [all(first_ok[: k + 1]) for k in range(3)]})
        rec.save(out / "gifs" / f"retry_t{i:02d}_{''.join('o' if o else 'x' for o in ok)}.gif")
        print(f"seq {i}: {''.join('o' if o else 'x' for o in ok)} attempts {[s['attempts'] for s in seq]}", flush=True)
    cum = np.array([s["cumulative"] for s in seqs], float).mean(0)
    cum0 = np.array([s["cumulative_first"] for s in seqs], float).mean(0)
    att = np.array([[st["attempts"] for st in s["stages"]] for s in seqs])
    res = {"ckpt": a.ckpt, "trigger": a.trigger, "max_attempts": a.max_attempts, "cumulative_success": cum.tolist(),
           "cumulative_first_attempt": cum0.tolist(), "mean_attempts_per_stage": att.mean(0).tolist(),
           "retried_stage_runs": int((att > 1).sum()), "n": len(seqs), "sequences": seqs}
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"[retry] cumulative {np.round(100 * cum, 1).tolist()} (first attempt only {np.round(100 * cum0, 1).tolist()}), "
          f"retried stage runs {res['retried_stage_runs']}", flush=True)
    rec.close()
    env.close()


if __name__ == "__main__":
    main()
