"""Collect scripted-expert demonstrations for one stacking stage into an HDF5 file.

Each episode: 270 control steps at 30 Hz with joint state, joint-target action and the
"front" (wrist) / "top" (overhead) 160x120 RGB images. Only episodes whose final bin pose
passes the stage success check are kept (like discarding failed teleoperation takes).

usage: python gen_data.py --stage 1 --episodes 100 --seed 1000
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

from expert import DESIGN, ScriptedExpert, stage2_target
from stack_env import BIN_H, BIN_NAMES, TARGETS, StackEnv, sample_scene


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", default="data")
    ap.add_argument("--dart-sigma", type=float, default=0.0,
                    help="DART: Ornstein-Uhlenbeck noise (rad) added to the executed arm targets; the "
                         "recorded action stays the clean expert target, so the demos show recoveries")
    ap.add_argument("--name", default=None, help="output file stem (default stage{N})")
    ap.add_argument("--dr", action="store_true", help="domain randomisation: lights, table colour, camera mounts, "
                    "distractors, wider pick range (+-28 mm, +-30 deg)")
    ap.add_argument("--dr-level", type=int, default=1, help="1 = v6 ranges, 2 = v7 (textures, tinted 0.25-2.5x light)")
    ap.add_argument("--clutter", action="store_true", help="physical clutter on the table: 0-8 objects, 12-35 mm "
                    "margin around the arm's path, a look-alike bin and a passing object each in 30%% of the "
                    "episodes; an episode where the expert moved any of it is discarded")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = args.name or f"stage{args.stage}"
    path = out / f"{stem}.hdf5"
    env = StackEnv(render=True, dr=args.dr, clutter=args.clutter)
    clut_rng = np.random.default_rng(args.seed + args.stage * 100000 + 13)
    vis_rng = np.random.default_rng(args.seed + args.stage * 100000 + 7)
    rng = np.random.default_rng(args.seed + args.stage * 100000)
    ex = ScriptedExpert(env, rng)
    home = ex.home_q()
    kept, tried, t0 = 0, 0, time.time()
    log = []
    with h5py.File(path, "w") as f:
        f.attrs["stage"] = args.stage
        f.attrs["fps"] = 30
        f.attrs["home"] = home
        f.attrs["dart_sigma"] = args.dart_sigma
        while kept < args.episodes:
            tried += 1
            spec = sample_scene(args.stage, rng, wide=args.dr)
            if DESIGN == "v8" and args.stage == 2:  # also show A left shifted towards B's slot (seen in chained runs)
                p, yaw = spec.placed["bin_a"]
                p = p.copy()
                p[1] = TARGETS[1][1] + rng.uniform(-0.008, 0.013)
                spec.placed["bin_a"] = (p, yaw)
            obs = env.reset(spec, home)
            vis = env.randomize(vis_rng, level=args.dr_level) if args.dr else None
            clut = None
            if args.clutter:
                clut = env.place_clutter(clut_rng, int(clut_rng.integers(0, 9)), float(clut_rng.uniform(0.012, 0.035)),
                                         decoy=bool(clut_rng.random() < 0.3), mover=bool(clut_rng.random() < 0.3))
            if args.dr or args.clutter:
                obs = env.observe()
            crefs = env.clutter_positions()
            refs = {n: env.bin_state(n).pos.copy() for n in BIN_NAMES[: args.stage - 1]}
            bs = env.bin_state(BIN_NAMES[args.stage - 1])
            a_now = env.bin_state("bin_a").pos
            tgt = {1: TARGETS[1], 2: stage2_target(a_now), 3: a_now + np.array([0, 0, BIN_H])}[args.stage]
            traj = ex.plan(env.qpos(), bs.pos, bs.yaw, tgt)
            S, A, F, T = [], [], [], []
            noise = np.zeros(6)
            for a in traj:
                S.append(obs["state"])
                A.append(a)
                F.append(obs["front"])
                T.append(obs["top"])
                if args.dart_sigma > 0:
                    noise[:5] += 0.1 * (-noise[:5]) + args.dart_sigma * rng.standard_normal(5)
                env.step(a + noise)
                obs = env.observe()
            env.settle(10)
            r = env.evaluate_stage(args.stage, ref_positions=refs)
            moved = env.clutter_moved(crefs)
            r["success"] = r["success"] and not moved
            log.append({"try": tried, "success": r["success"], "xy_err_mm": r["xy_err_mm"], "ik_err_mm": ex.last_ik_err * 1000,
                        **({"vis": vis} if vis else {}), **({"clutter": clut, "clutter_moved": moved} if args.clutter else {})})
            if not r["success"]:
                continue
            g = f.create_group(f"episode_{kept:04d}")
            g.create_dataset("state", data=np.array(S, np.float32))
            g.create_dataset("action", data=np.array(A, np.float32))
            g.create_dataset("front", data=np.array(F, np.uint8), compression="lzf", chunks=(1, *F[0].shape))
            g.create_dataset("top", data=np.array(T, np.uint8), compression="lzf", chunks=(1, *T[0].shape))
            g.attrs["xy_err_mm"] = r["xy_err_mm"]
            kept += 1
            if kept % 10 == 0:
                print(f"stage {args.stage}: kept {kept}/{tried} ({time.time() - t0:.0f}s)", flush=True)
        f.attrs["tried"] = tried
        f.attrs["kept"] = kept
    summary = {"stage": args.stage, "design": DESIGN, "dr": args.dr, "dr_level": args.dr_level if args.dr else 0, "clutter": args.clutter, "dart_sigma": args.dart_sigma, "kept": kept, "tried": tried, "expert_success_rate": kept / tried,
               "mean_xy_err_mm": float(np.mean([l["xy_err_mm"] for l in log if l["success"]])),
               "seconds": time.time() - t0}
    (out / f"{stem}_gen.json").write_text(json.dumps({"summary": summary, "log": log}, indent=1))
    print(json.dumps(summary))
    env.close()


if __name__ == "__main__":
    main()
