"""Chained success in the factory_vis / factory_max conditions by the look each trial drew (table kind, light).

The looks are not stored in the clutter results, but they are drawn from a fixed seed per trial (clutter_eval.py:
VIS_SEED0 + 90000 + trial, level 2), so they are drawn again here.

usage: python factory_vis_diag.py v11_n1500 [v12_n1800 ...]  -> results/clutter/<tag>/vis_diag.json
"""

import json
import sys
from pathlib import Path

import numpy as np

from clutter_eval import VIS_SEED0
from stack_env import StackEnv

env = StackEnv(render=False, clutter=True, factory=True)
for tag in sys.argv[1:]:
    out = {}
    for cond in ("factory_vis", "factory_max"):
        r = json.loads(Path(f"results/clutter/{tag}/{cond}.json").read_text())["chained"]
        rows = []
        for s in r["sequences"]:
            v = env.randomize(np.random.default_rng(VIS_SEED0 + 90000 + s["trial"]), level=2)
            rows.append({"trial": s["trial"], "ok": all(x["success"] for x in s["stages"]), "table": v["table"],
                         "key_light": v["key_light"], "tinted": v["tint"] != [1.0, 1.0, 1.0]})
        by = {}
        for kind, sel in (("plain", lambda x: x["table"] in ("plain", "wood") or isinstance(x["table"], list)),
                          ("patterned", lambda x: x["table"] in ("stripes", "checker", "blotches")),
                          ("dark (key light < 0.8)", lambda x: x["key_light"] < 0.8),
                          ("tinted", lambda x: x["tinted"])):
            m = [x["ok"] for x in rows if sel(x)]
            by[kind] = {"n": len(m), "success": round(100 * float(np.mean(m)), 1) if m else None}
        out[cond] = {"by_look": by, "trials": rows}
        print(tag, cond, {k: f"{v['success']}% (n={v['n']})" for k, v in by.items()})
    Path(f"results/clutter/{tag}/vis_diag.json").write_text(json.dumps(out, indent=1))
