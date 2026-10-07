"""Merge evaluation parts run with eval_act.py --start/--trials into one results.json (same format).
usage: python merge_eval.py --out results/eval/v5_n100_x200 results/eval/v5_n100_p0 results/eval/v5_n100_p1 ..."""
import argparse
import json
from pathlib import Path

import numpy as np

from eval_act import follow_slopes

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("parts", nargs="+")
a = ap.parse_args()
rs = [json.loads((Path(p) / "results.json").read_text()) for p in a.parts]
res = {k: rs[0][k] for k in ("ckpt", "temporal_ensemble", "n_action_steps") if k in rs[0]}
res["parts"] = a.parts
res["devices"] = sorted({r.get("device", "?") for r in rs})
per = {}
for st in ("1", "2", "3"):
    tr = sorted((t for r in rs for t in r["per_stage"][st]["trials"]), key=lambda t: t["trial"])
    per[st] = {"success_rate": float(np.mean([t["success"] for t in tr])), "n": len(tr),
               "xy_err_mm_mean_success": float(np.mean([t["xy_err_mm"] for t in tr if t["success"]] or [np.nan])),
               "follow": follow_slopes(tr), "trials": tr}
seqs = sorted((s for r in rs for s in r["chained"]["sequences"]), key=lambda s: s["trial"])
cum = np.array([s["cumulative"] for s in seqs], float).mean(0)
res["per_stage"] = per
res["chained"] = {"cumulative_success": cum.tolist(), "n": len(seqs), "sequences": seqs}
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
(out / "results.json").write_text(json.dumps(res, indent=1, default=float))
print({st: f"{100 * v['success_rate']:.1f}% ({v['n']})" for st, v in per.items()}, "chained", np.round(100 * cum, 1).tolist(), len(seqs))
