"""Check that the trained ACT policies act on what the cameras show (imitation learning, not trajectory replay).

Same per-stage scenes and success test as eval_act.py; the policy is run twice per scene:
  normal : real camera images
  blind  : both camera images replaced by black frames (joint state unchanged)
If the policy only replayed a memorised motion, blinding would not matter; if it localises the bin from the
images, success should collapse and the grasp should no longer follow the bin.
usage: python verify_il.py --ckpt S1 S2 S3 --stages 1 2 --trials 10 --out results/verify_il.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from eval_act import EVAL_SEED0, Recorder, load_policy, run_stage
from expert import ScriptedExpert
from stack_env import BIN_NAMES, StackEnv, sample_scene


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs=3, required=True)
    ap.add_argument("--stages", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--out", default="results/verify_il.json")
    a = ap.parse_args()
    env = StackEnv(render=True)
    home = ScriptedExpert(env).home_q()
    pols = [load_policy(p, "cpu", None) for p in a.ckpt]  # same settings as robust_eval.py
    rec = Recorder(env, enabled=False)
    real_observe = env.observe

    def blind_observe():
        o = real_observe()
        return {**o, "front": np.zeros_like(o["front"]), "top": np.zeros_like(o["top"])}

    out = {"ckpt": a.ckpt, "trials": a.trials}
    for stage in a.stages:
        for mode in ("normal", "blind"):
            env.observe = real_observe if mode == "normal" else blind_observe
            rs = []
            for i in range(a.trials):
                env.reset(sample_scene(stage, np.random.default_rng(EVAL_SEED0 + 1000 * stage + i)), home)
                refs = {n: env.bin_state(n).pos.copy() for n in BIN_NAMES[: stage - 1]}
                info = run_stage(env, pols[stage - 1], home, "cpu", rec)
                r = env.evaluate_stage(stage, ref_positions=refs)
                gs = info["grasp_site"]
                miss = None if gs is None else 1000 * float(np.linalg.norm(np.array(gs[:2]) - np.array(info["bin0"][:2])))
                rs.append({"success": r["success"], "xy_err_mm": r["xy_err_mm"], "grasp_to_bin_mm": miss})
            k = sum(r["success"] for r in rs)
            g = [r["grasp_to_bin_mm"] for r in rs if r["grasp_to_bin_mm"] is not None]
            out[f"s{stage}_{mode}"] = {"success": k, "n": len(rs), "grasp_to_bin_mm_mean": float(np.mean(g)) if g else None,
                                       "trials": rs}
            print(f"stage {stage} {mode:6s}: {k}/{len(rs)} success, grasp-to-bin mean "
                  f"{np.mean(g):.1f} mm" if g else f"stage {stage} {mode}: {k}/{len(rs)} success, no grasp", flush=True)
    env.observe = real_observe
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
