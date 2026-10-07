"""Summarise evaluation results: success table, failure modes, bin-following slopes, learning curve.

usage: python analyze.py            -> prints a markdown report and writes results/summary.json
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
EVAL = ROOT / "results" / "eval"
CONDS = ["v1_n100", "n100", "n200",  # v1, v2 designs (robot-facing wall)
         "v4_n100", "v4_n200",  # v4: one fixed grasp wall
         "v5_n100", "v5_n200", "v5_n500", "v5_n1000", "v5_dart1000", "v5_dart200", "v5_n1000_te",  # final design
         "v6_n1000", "v6c_n1000", "v5c_n200",  # domain randomisation / brightness normalisation + CLAHE
         "v5w_n200", "v5t_n200"]  # camera ablation: wrist only / top only


def load(name):
    p = EVAL / name / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


def failure_mode(t: dict, stage: int) -> str:
    """Classify one evaluated run."""
    if t["success"]:
        return "success"
    if t.get("disturbed"):
        return "pushed_previous"
    if t.get("grasp_site") is None:
        return "no_grasp"
    pos = t.get("bin_pos")
    if t["tilt_deg"] >= 10:
        return "tilted"
    if pos is not None and abs(t["z_err_mm"]) >= 8:
        # bin still on the scale (~z 0.051) or dropped somewhere else
        return "not_placed" if pos[1] < -0.05 else "wrong_height"
    return "misplaced"


def summarise(name: str, r: dict) -> dict:
    out = {"name": name}
    if "per_stage" in r:
        for s in ("1", "2", "3"):
            ps = r["per_stage"][s]
            modes = {}
            for t in ps["trials"]:
                m = failure_mode(t, int(s))
                modes[m] = modes.get(m, 0) + 1
            out[f"stage{s}"] = {"success": ps["success_rate"], "n": ps["n"], "modes": modes,
                                "xy_err_mm": ps["xy_err_mm_mean_success"], "follow": ps.get("follow", {})}
    if "chained" in r:
        ch = r["chained"]
        first_fail = {"1": 0, "2": 0, "3": 0, "none": 0}
        for seq in ch["sequences"]:
            k = next((i for i, ok in enumerate(seq["cumulative"]) if not ok), None)
            first_fail["none" if k is None else str(k + 1)] += 1
        out["chained"] = {"cumulative": ch["cumulative_success"], "n": ch["n"], "first_failure_stage": first_fail}
        # conditional success of stage k given stages < k succeeded
        cond = []
        for k in range(3):
            prev = [s for s in ch["sequences"] if all(s["cumulative"][:k]) if k == 0 or s["cumulative"][k - 1]]
            ok = [s for s in prev if s["stages"][k]["success"]]
            cond.append(len(ok) / len(prev) if prev else float("nan"))
        out["chained"]["conditional"] = cond
    return out


def learning_curve(name: str) -> dict:
    pts = {}
    for d in sorted(EVAL.glob(f"{name}_ckpt*")):
        r = json.loads((d / "results.json").read_text()) if (d / "results.json").exists() else None
        if r and "per_stage" in r:
            step = int(d.name.split("ckpt")[-1])
            pts[step] = [r["per_stage"][s]["success_rate"] for s in ("1", "2", "3")]
    full = load(name)
    if full and "per_stage" in full:
        meta = ROOT / "runs" / f"s1_{name}" / "train_meta.json"
        steps = json.loads(meta.read_text())["steps"] if meta.exists() else None
        if steps:
            pts[steps] = [full["per_stage"][s]["success_rate"] for s in ("1", "2", "3")]
    return dict(sorted(pts.items()))


def main() -> None:
    summary = {}
    lines = ["| 조건 | 1층 | 옆 | 2층 | 연속(1→2→3) | 따라가기 기울기 x/y (1층) |", "|---|---|---|---|---|---|"]
    for c in CONDS:
        r = load(c)
        if not r:
            continue
        s = summarise(c, r)
        s["learning_curve"] = learning_curve(c)
        summary[c] = s
        cells = []
        for k in ("stage1", "stage2", "stage3"):
            cells.append(f"{100 * s[k]['success']:.0f}%" if k in s else "-")
        ch = s.get("chained", {}).get("cumulative")
        cells.append(" → ".join(f"{100 * v:.0f}" for v in ch) if ch else "-")
        f = s.get("stage1", {}).get("follow", {})
        cells.append(f"{f['slope_x']:.2f} / {f['slope_y']:.2f}" if f else "-")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    print("\n".join(lines))
    for c, s in summary.items():
        print(f"\n### {c}")
        for k in ("stage1", "stage2", "stage3"):
            if k in s:
                print(f"- {k}: {s[k]['modes']}, 평균 오차 {s[k]['xy_err_mm']:.1f}mm")
        if "chained" in s:
            print(f"- 연속: 처음 실패한 단계 {s['chained']['first_failure_stage']}, "
                  f"조건부 성공 {[round(v, 2) for v in s['chained']['conditional']]}")
        if s["learning_curve"]:
            print(f"- 학습 곡선: {s['learning_curve']}")
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
