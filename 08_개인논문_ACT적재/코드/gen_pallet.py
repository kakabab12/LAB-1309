"""Demonstrations for one slot of the 2x2 x 2-layer pallet (final design, layout x0 0.20 / y0 0.03).

Scene for slot k: the bins of slots 0..k-1 already stand at their targets with placement-like noise (+-4 mm,
+-4 deg; upper-layer bins on top of the actual lower bin), the bin for slot k is on the scale at a random
pose (+-20 mm, +-20 deg). The expert (pallet_grid.GridExpert) moves it; only successful episodes are kept.
Same HDF5 layout as gen_data.py, so convert_jpeg.py / train_act.py work unchanged.
usage: ACT_DESIGN=v4 PALLET_VIA=far python gen_pallet.py --slot 0 --episodes 100 --seed 21000 --name pallet_s1
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import h5py
import numpy as np

import pallet_grid as PG
import stack_env as SE

LAYOUT = dict(rows=2, cols=2, layers=2, x0=float(os.environ.get("PALLET_X0", "0.20")),
              y0=float(os.environ.get("PALLET_Y0", "0.03")))  # PALLET_X0/Y0/GAP_Y: pallet layout v2 (10/9)
NOISE_XY, NOISE_YAW = 0.003, math.radians(2)  # small enough that neighbours (8 mm apart) never overlap


def scene(slots, k, rng):
    """SceneSpec for slot k (0-based): earlier slots placed with noise, slot k's bin on the scale."""
    placed = {}
    for j in range(k):
        s = slots[j]
        if s["below"] is None:
            xy = s["xy"] + rng.uniform(-NOISE_XY, NOISE_XY, size=2)
            z = SE.BIN_H / 2
        else:
            lower = placed[slots[s["below"]]["name"]][0]
            xy = lower[:2] + rng.uniform(-NOISE_XY / 2, NOISE_XY / 2, size=2)
            z = lower[2] + SE.BIN_H + 0.0005  # rest on the lower bin's rim, not inside it
        placed[s["name"]] = (np.array([xy[0], xy[1], z]), float(rng.uniform(-NOISE_YAW, NOISE_YAW)))
    xy, yaw = SE.sample_pick(rng)
    return SE.SceneSpec(stage=k + 1, new_bin_xy=xy, new_bin_yaw=yaw, placed=placed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", type=int, required=True, help="0-based slot index in placement order")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=21000)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    slots = PG.configure(**LAYOUT)
    env = PG.make_env(slots, render=True)
    rng = np.random.default_rng(a.seed + 1000 * a.slot)
    ex = PG.GridExpert(env, rng)
    home = ex.home_q()
    k, s = a.slot, slots[a.slot]
    out = Path(a.out)
    path = out / f"{a.name}.hdf5"
    kept, tried, log, t0 = 0, 0, [], time.time()
    with h5py.File(path, "w") as f:
        f.attrs.update(slot=k, layer=s["layer"], row=s["row"], col=s["col"], fps=30, home=home)
        while kept < a.episodes:
            tried += 1
            obs = env.reset(scene(slots, k, rng), home)
            refs = {t["name"]: env.bin_state(t["name"]).pos.copy() for t in slots[:k]}
            bs = env.bin_state(s["name"])
            tgt = PG.slot_target(env, slots, k)
            traj = ex.plan(env.qpos(), bs.pos, bs.yaw, tgt, upper=s["layer"] > 0, far=s["row"] == LAYOUT["rows"] - 1)
            S, A, F, T = [], [], [], []
            for act in traj:
                S.append(obs["state"])
                A.append(act)
                F.append(obs["front"])
                T.append(obs["top"])
                env.step(act)
                obs = env.observe()
            env.settle(10)
            r = PG.evaluate(env, slots, k, refs, tgt)
            log.append({"try": tried, "success": r["success"], "xy_err_mm": r["xy_err_mm"]})
            if not r["success"]:
                continue
            g = f.create_group(f"episode_{kept:04d}")
            g.create_dataset("state", data=np.array(S, np.float32))
            g.create_dataset("action", data=np.array(A, np.float32))
            g.create_dataset("front", data=np.array(F, np.uint8), compression="lzf", chunks=(1, *F[0].shape))
            g.create_dataset("top", data=np.array(T, np.uint8), compression="lzf", chunks=(1, *T[0].shape))
            kept += 1
            if kept % 10 == 0:
                print(f"slot {k}: kept {kept}/{tried} ({time.time() - t0:.0f}s)", flush=True)
        f.attrs["tried"], f.attrs["kept"] = tried, kept
    summary = {"slot": k, "layer": s["layer"], "row": s["row"], "col": s["col"], "kept": kept, "tried": tried,
               "expert_success_rate": kept / tried, "seconds": time.time() - t0, "layout": LAYOUT}
    (out / f"{a.name}_gen.json").write_text(json.dumps({"summary": summary, "log": log}, indent=1))
    print(json.dumps(summary))
    env.close()


if __name__ == "__main__":
    main()
