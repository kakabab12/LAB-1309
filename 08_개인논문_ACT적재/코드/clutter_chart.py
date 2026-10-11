"""Bar chart of the clutter / factory evaluation: chained 3-stage success per condition and model
(results/clutter/<tag>/<cond>.json, and the integrated bin model if it is evaluated) -> results/clutter_chained.png.
Models that are not evaluated yet are left out, so it can be re-run as results come in.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 10.5})

ROOT = Path(__file__).resolve().parent
CONDS = [("none", "변화 없음"), ("all", "잡동사니 8개\n+닮은 통+이동"), ("tight", "잡동사니\n팔 경로 5mm"),
         ("factory", "공장형"), ("factory_vis", "공장형\n+외관"), ("factory_max", "공장형 최대")]
MODELS = [("v5_n100", "규칙 시연 100회", "#9e9e9e"), ("v6_n1000", "+ 외관 무작위화 (1000회)", "#64b5f6"),
          ("v9_n1000", "+ 잡동사니 시연 (1000회)", "#1565c0"), ("v11_n1500", "+ 공장 시연 500회", "#2e7d32"),
          ("int120_n1800_bin_orig", "세 물체 통합 정책 (통, 원래 배치)", "#e65100")]  # intermediate rounds: journal tables
if all((ROOT / "results" / "clutter" / "v12_n1800" / f"{c}.json").exists() for c, _ in CONDS):  # same rule as the paper
    MODELS[3] = ("v12_n1800", "+ 공장 시연 800회 (외관 포함)", "#2e7d32")


def load(tag: str, cond: str):
    for p in (ROOT / "results" / "clutter" / tag / f"{cond}.json",):
        if p.exists():
            d = json.loads(p.read_text())
            return 100 * d["chained"]["cumulative_success"][2], d["chained"]["n"]
    return None


def main() -> None:
    have = [(t, lab, c) for t, lab, c in MODELS if any(load(t, k) for k, _ in CONDS)]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    w = 0.8 / len(have)
    for j, (t, lab, col) in enumerate(have):
        for i, (k, _) in enumerate(CONDS):
            r = load(t, k)
            x = i + (j - (len(have) - 1) / 2) * w
            if r is None:
                if not (t.startswith("int") and k == "tight"):  # the integrated models are not run on "tight"
                    ax.text(x, 2, "평가 중", ha="center", va="bottom", fontsize=6, color=col, rotation=90)
                continue
            ax.bar(x, max(r[0], 0.6), w * 0.92, color=col, label=lab if i == 0 else None)
            ax.text(x, r[0] + 1.5, f"{r[0]:.0f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(CONDS)), [n for _, n in CONDS])
    ax.set_ylim(0, 110)
    ax.set_ylabel("연속 3단계 성공률 (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    n = load(have[0][0], "none")[1]
    ax.set_title(f"잡동사니·공장 환경 (시뮬레이션, 조건마다 {n}회, 1층→옆→2층 모두 성공해야 성공)", fontsize=10.5)
    fig.tight_layout()
    out = ROOT / "results" / "clutter_chained.png"
    fig.savefig(out, dpi=200)
    print(out, [t for t, _, _ in have])


if __name__ == "__main__":
    main()
