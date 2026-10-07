"""Where does the policy close the gripper relative to the expert's grasp point?

For every per-stage eval trial, rebuilds the scene from its seed (bin yaw), takes the facing-wall normal
and splits the closing position into: normal offset (negative = toward/onto the wall), tangential offset
and height offset, all relative to the demonstration design's grasp point. Compares the offsets with the
design's tolerance window (grasp_tolerance_all.json).

usage: python grasp_offsets.py n100 n200 [--clear 0.009] [--check-expert]
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

import expert as E
from stack_env import BIN_H, BIN_W, StackEnv, sample_scene

EVAL_SEED0 = 500000


def offsets(name: str, clear: float) -> dict:
    r = json.loads(Path(f"results/eval/{name}/results.json").read_text())
    out = {}
    for stage in ("1", "2", "3"):
        rows = []
        for t in r["per_stage"][stage]["trials"]:
            spec = sample_scene(int(stage), np.random.default_rng(EVAL_SEED0 + 1000 * int(stage) + t["trial"]))
            b0 = np.array(t["bin0"])
            n = E.ScriptedExpert.facing_wall_normal(b0[:2], spec.new_bin_yaw)
            row = {"trial": t["trial"], "success": t["success"]}
            if t.get("grasp_site") is not None:
                d = np.array(t["grasp_site"]) - b0
                row["normal_mm"] = 1000 * (d[:2] @ n[:2] - (BIN_W / 2 + clear))
                row["tangent_mm"] = 1000 * (d[0] * -n[1] + d[1] * n[0])
                row["height_mm"] = 1000 * (d[2] - (BIN_H / 2 - E.ScriptedExpert.GRASP_DEPTH))
            rows.append(row)
        out[stage] = rows
    return out


def describe(name: str, res: dict) -> None:
    print(f"== {name}")
    for stage, rows in res.items():
        g = [r for r in rows if "normal_mm" in r]
        ok = np.array([r["normal_mm"] for r in g if r["success"]])
        bad = [r for r in rows if not r["success"]]
        print(f" stage {stage}: normal offset ok mean {ok.mean():+.1f} sd {ok.std():.1f} "
              f"[{ok.min():+.1f}, {ok.max():+.1f}] mm  | tangent sd {np.std([r['tangent_mm'] for r in g]):.1f}")
        for r in bad:
            if "normal_mm" in r:
                print(f"   fail t{r['trial']:02d}: normal {r['normal_mm']:+.1f} tangent {r['tangent_mm']:+.1f} "
                      f"height {r['height_mm']:+.1f} mm")
            else:
                print(f"   fail t{r['trial']:02d}: never closed")


def check_expert(clear: float, k: int = 4) -> None:
    """The reference must match: the expert itself should land at ~0 mm."""
    env = StackEnv(render=False)
    ex = E.ScriptedExpert(env, np.random.default_rng(0))
    ex.CLEAR = clear
    home = ex.home_q()
    for stage in (1, 2, 3):
        for i in range(k):
            spec = sample_scene(stage, np.random.default_rng(EVAL_SEED0 + 1000 * stage + i))
            env.reset(spec, home)
            bs = env.bin_state(env.active_bin)
            b0 = bs.pos.copy()
            from stack_env import TARGETS
            traj = ex.plan(env.qpos(), bs.pos, bs.yaw, TARGETS[stage], jitter=False)
            opened, site = False, None
            for a in traj:
                env.step(a)
                opened |= a[5] > 0.1
                if site is None and opened and a[5] < 0.0:
                    site = env.d.site_xpos[env.site].copy()
            n = E.ScriptedExpert.facing_wall_normal(b0[:2], spec.new_bin_yaw)
            d = site - b0
            print(f" expert s{stage} t{i}: normal {1000 * (d[:2] @ n[:2] - (BIN_W / 2 + clear)):+.1f} "
                  f"height {1000 * (d[2] - (BIN_H / 2 - E.ScriptedExpert.GRASP_DEPTH)):+.1f} mm "
                  f"(yaw diff {math.degrees(bs.yaw - spec.new_bin_yaw):+.2f} deg)")
    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--clear", type=float, default=0.009)
    ap.add_argument("--check-expert", action="store_true")
    a = ap.parse_args()
    if a.check_expert:
        check_expert(a.clear)
    allres = {}
    for nm in a.names:
        allres[nm] = offsets(nm, a.clear)
        describe(nm, allres[nm])
    if allres:
        Path("results/grasp_offsets.json").write_text(json.dumps(allres, indent=1))
