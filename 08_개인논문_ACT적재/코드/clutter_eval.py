"""Stacking policies on a cluttered table: other (physical) objects stand around the cell.

Same protocols, seeds and success test as eval_act.py, with clutter put on the table after each reset
(StackEnv.place_clutter: only where the expert's arm never came low, see clutter_map.py). The clutter scene of a
trial is the same for every policy. Success additionally needs every clutter object to stay within 10 mm of where
it stood when the stage started (the arm must not push or knock anything over), so the test is stricter, never
looser, than the plain one. The plain success is logged as well.

Conditions (objects, margin around the arm's path, look-alike bin, passing object, randomised looks):
  none         no clutter (reference, same scenes)
  sparse       4 objects, 30 mm margin
  dense        8 objects, 12 mm margin
  decoy        a non-blue empty copy of the bin + 4 objects, 30 mm margin
  moving       4 objects + an object passing by (4-10 cm/s, back and forth) while the arm works
  dense_decoy  look-alike + 8 objects, 12 mm margin
  all          look-alike + 8 objects + passing object, 12 mm margin
  all_vis      as all, plus random light / table colour / camera mounting (v6 domain-randomisation ranges)
  tight        look-alike + 8 objects + passing object, 5 mm margin (closer than any training scene)
Factory cell (factory.py: conveyor with running items, rack of other bins, fence posts, control box, overhead cables
over the top camera, hazard tapes, 2 look-alikes, 2 passing objects, flickering light; touching a structure = failure):
  factory      14 table parts from 16 kinds, 10 mm margin
  factory_vis  as factory, plus wide random looks (light 0.25-2.5x + tint, table colour / texture: v7 ranges)
  factory_max  16 parts, 5 mm margin, wide random looks

usage: python clutter_eval.py --ckpt S1 S2 S3 --tag v5_n100 --conds none sparse dense decoy --trials 50
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval_act import EVAL_SEED0, Recorder, load_policy, run_stage
from expert import ScriptedExpert
from stack_env import BIN_NAMES, StackEnv, sample_pick, sample_scene

CONDS = {  # objects, margin, look-alike, passing object, randomised looks
    "none": (0, 0.030, False, False, False), "sparse": (4, 0.030, False, False, False),
    "dense": (8, 0.012, False, False, False), "decoy": (4, 0.030, True, False, False),
    "moving": (4, 0.030, False, True, False), "dense_decoy": (8, 0.012, True, False, False),
    "all": (8, 0.012, True, True, False), "all_vis": (8, 0.012, True, True, True), "tight": (8, 0.005, True, True, False),
    "factory": (14, 0.010, True, True, False), "factory_vis": (14, 0.010, True, True, 2), "factory_max": (16, 0.005, True, True, 2)}
FACTORY = {"factory", "factory_vis", "factory_max"}
VIS_SEED0 = 660000
CLUTTER_SEED0 = 770000


def stage_result(env: StackEnv, stage: int, refs: dict, crefs: dict) -> dict:
    r = env.evaluate_stage(stage, ref_positions=refs)
    moved = env.clutter_moved(crefs)
    r["clutter_moved"] = moved
    r["struct_hit"] = bool(env.struct_hit)
    r["success_plain"] = r["success"]
    r["success"] = bool(r["success"] and not moved and not env.struct_hit)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs=3, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--conds", nargs="+", default=list(CONDS))
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--mode", choices=["per_stage", "chained", "both"], default="both")
    ap.add_argument("--n-action-steps", type=int, default=None)
    ap.add_argument("--gifs", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/clutter")
    ap.add_argument("--factory-env", action="store_true", help="factory scene file for every condition (structures parked)")
    a = ap.parse_args()
    out = Path(a.out) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    env = StackEnv(render=True, clutter=True, factory=bool(FACTORY & set(a.conds)) or a.factory_env)
    home = ScriptedExpert(env).home_q()
    pols = [load_policy(p, a.device, None, a.n_action_steps) for p in a.ckpt]
    rec = Recorder(env, enabled=a.gifs > 0)
    for cond in a.conds:
        f = out / f"{cond}.json"
        if f.exists():
            continue
        n, margin, decoy, mover, vis = CONDS[cond]
        res = {"cond": cond, "ckpt": a.ckpt, "n_objects": n, "margin_m": margin, "decoy": decoy, "mover": mover, "vis": vis}

        def setup(seed: int) -> dict:
            if vis:
                env.randomize(np.random.default_rng(VIS_SEED0 + seed), level=int(vis))
            else:
                env.reset_visuals()
            if cond in FACTORY:
                import factory
                return factory.place_factory(env, np.random.default_rng(CLUTTER_SEED0 + seed), n, margin)
            return env.place_clutter(np.random.default_rng(CLUTTER_SEED0 + seed), n, margin, decoy, mover=mover)

        if a.mode in ("per_stage", "both"):
            per = {}
            for stage in (1, 2, 3):
                rs = []
                for i in range(a.trials):
                    rng = np.random.default_rng(EVAL_SEED0 + 1000 * stage + i)
                    env.reset(sample_scene(stage, rng), home)
                    placed = setup(1000 * stage + i)
                    refs = {b: env.bin_state(b).pos.copy() for b in BIN_NAMES[: stage - 1]}
                    crefs = env.clutter_positions()
                    rec.enabled = i < a.gifs
                    info = run_stage(env, pols[stage - 1], home, a.device, rec)
                    r = stage_result(env, stage, refs, crefs)
                    r.update(info, trial=i, clutter=placed)
                    rs.append(r)
                    rec.save(out / "gifs" / f"{cond}_s{stage}_t{i:02d}_{'ok' if r['success'] else 'fail'}.gif")
                per[stage] = {"success_rate": float(np.mean([r["success"] for r in rs])),
                              "success_plain": float(np.mean([r["success_plain"] for r in rs])),
                              "clutter_moved_rate": float(np.mean([bool(r["clutter_moved"]) for r in rs])),
                              "n": len(rs), "trials": rs}
                print(f"{a.tag} {cond} stage {stage}: {100 * per[stage]['success_rate']:.0f}% "
                      f"(plain {100 * per[stage]['success_plain']:.0f}%, clutter moved "
                      f"{100 * per[stage]['clutter_moved_rate']:.0f}%)", flush=True)
            res["per_stage"] = per
        if a.mode in ("chained", "both"):
            seqs = []
            for i in range(a.trials):
                rng = np.random.default_rng(EVAL_SEED0 + 90000 + i)
                env.reset(sample_scene(1, rng), home)
                placed = setup(90000 + i)
                rec.enabled = i < a.gifs
                seq = []
                for stage in (1, 2, 3):
                    if stage > 1:
                        xy, yaw = sample_pick(rng)
                        env.spawn_next_bin(stage, xy, yaw)
                    refs = {b: env.bin_state(b).pos.copy() for b in BIN_NAMES[: stage - 1]}
                    crefs = env.clutter_positions()
                    env.struct_hit = False  # per stage (factory scenes)
                    info = run_stage(env, pols[stage - 1], home, a.device, rec)
                    r = stage_result(env, stage, refs, crefs)
                    r.update(info)
                    seq.append(r)
                ok = [s["success"] for s in seq]
                seqs.append({"trial": i, "clutter": placed, "stages": seq, "cumulative": [all(ok[: k + 1]) for k in range(3)]})
                rec.save(out / "gifs" / f"{cond}_chained_t{i:02d}_{''.join('o' if o else 'x' for o in ok)}.gif")
            cum = np.array([s["cumulative"] for s in seqs], float).mean(0)
            res["chained"] = {"cumulative_success": cum.tolist(), "n": len(seqs), "sequences": seqs}
            print(f"{a.tag} {cond} chained: {' -> '.join(f'{100 * c:.0f}%' for c in cum)}", flush=True)
        f.write_text(json.dumps(res, default=float))
    rec.close()
    env.close()


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
