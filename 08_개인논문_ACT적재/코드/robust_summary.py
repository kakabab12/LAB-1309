"""Collect robust_eval.py results into one table (results/robust_summary.json + markdown on stdout).

For every model tag in results/robust/<tag>/<cond>.json: per-stage success and the mean over the stages
that the condition applies to. "chain_est" = product of the three per-stage rates (an estimate of the
3-stage sequence success; conditions that skip a stage use the base rate for it).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COND_ORDER = ["base", "pos_out", "yaw_out", "placed_out", "dark", "bright", "light_side", "table_gray", "table_dark",
              "cam_top", "cam_wrist", "distractor", "mass_100g", "mass_200g", "delay_67ms", "delay_133ms"]
COND_KO = {"base": "기준 (변화 없음)", "pos_out": "상자 위치 범위 밖", "yaw_out": "상자 회전 범위 밖",
           "placed_out": "아래 상자 어긋남", "dark": "어둡게 50%", "bright": "밝게 160%", "light_side": "옆 조명",
           "table_gray": "책상 회색", "table_dark": "책상 짙은 갈색", "cam_top": "상단 카메라 틀어짐",
           "cam_wrist": "손목 카메라 틀어짐", "distractor": "주변 물건", "mass_100g": "상자 무게 100g",
           "mass_200g": "상자 무게 200g", "delay_67ms": "관측 지연 67ms", "delay_133ms": "관측 지연 133ms"}


def load_all() -> dict:
    out = {}
    for d in sorted((ROOT / "results" / "robust").glob("*/")):
        rows = {}
        for c in COND_ORDER:
            f = d / f"{c}.json"
            if f.exists():
                ps = json.loads(f.read_text())["per_stage"]
                rows[c] = {int(s): v["success_rate"] for s, v in ps.items()}
        if rows:
            out[d.name] = rows
    return out


def summarise(res: dict) -> dict:
    summ = {}
    for tag, rows in res.items():
        base = rows.get("base", {})
        summ[tag] = {}
        for c, st in rows.items():
            mean = sum(st.values()) / len(st)
            full = {s: st.get(s, base.get(s)) for s in (1, 2, 3)}
            chain = None if None in full.values() else full[1] * full[2] * full[3]
            summ[tag][c] = {"per_stage": st, "mean": mean, "chain_est": chain}
    return summ


if __name__ == "__main__":
    s = summarise(load_all())
    (ROOT / "results" / "robust_summary.json").write_text(json.dumps(s, indent=1))
    tags = list(s)
    print("| 조건 | " + " | ".join(tags) + " |")
    print("|---|" + "---|" * len(tags))
    for c in COND_ORDER:
        cells = []
        for t in tags:
            v = s[t].get(c)
            cells.append("—" if v is None else
                         " / ".join(f"{100 * v['per_stage'][k]:.0f}" for k in sorted(v["per_stage"])) +
                         (f" (연속 추정 {100 * v['chain_est']:.0f})" if v["chain_est"] is not None else ""))
        print(f"| {COND_KO[c]} | " + " | ".join(cells) + " |")
