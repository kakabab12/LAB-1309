"""Robustness figure: mean per-stage success under each environment change, one marker per model.

Dot plot (conditions as rows) so several models fit a single column; reads results/robust_summary.json
(python robust_summary.py). Models shown: whichever of MODELS exist.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robust_summary import COND_KO, COND_ORDER  # noqa: E402

font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 6.5, "axes.linewidth": 0.6})
INK2, GRID = "#52514e", "#e4e3df"
MODELS = [  # tag, label, colour, marker
    ("v4_n100", "기존 (시연 100)", "#9a9994", "o"),
    ("v5_n1000", "기존 (시연 1000)", "#eb6834", "s"),
    ("v6_n1000", "무작위화", "#2a78d6", "^"),
    ("v6c_n1000", "무작위화 + 정규화·CLAHE", "#1baf7a", "D"),
]


def main(out: Path) -> None:
    p = ROOT / "results" / "robust_summary.json"
    if not p.exists():
        print("run robust_summary.py first")
        return
    s = json.loads(p.read_text())
    models = [m for m in MODELS if m[0] in s]
    if not models:
        print("no robustness results yet")
        return
    conds = [c for c in COND_ORDER if any(c in s[m[0]] for m in models)]
    fig, ax = plt.subplots(figsize=(3.3, 0.22 * len(conds) + 0.75), dpi=300)
    for k, (tag, label, col, mk) in enumerate(models):
        xs, ys = [], []
        for i, c in enumerate(conds):
            if c in s[tag]:
                xs.append(100 * s[tag][c]["mean"])
                ys.append(i + (k - (len(models) - 1) / 2) * 0.13)
        ax.scatter(xs, ys, s=12, marker=mk, color=col, edgecolor="white", linewidth=0.4, zorder=3, label=label)
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels([COND_KO[c] for c in conds])
    ax.invert_yaxis()
    ax.set_xlim(-3, 103)
    ax.set_xlabel("단계별 평균 성공률 (%)", color=INK2)
    ax.grid(axis="x", color=GRID, lw=0.5)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0, colors=INK2)
    ax.tick_params(axis="x", colors=INK2, length=2)
    ax.legend(frameon=False, fontsize=5.8, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2,
              handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout(pad=0.3)
    fig.savefig(out, dpi=300)
    print("saved", out)


if __name__ == "__main__":
    main(ROOT / "paper" / "fig_robust.png")
