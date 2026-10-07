"""Failure modes per demonstration design (per-stage evaluation, 3 x 50 trials, 100 demonstrations).

Stacked horizontal bars: success / no grasp (bin never left the scale) / tilted / misplaced / pushed an earlier
bin. Shows each design change removing one failure type. Reads results/summary.json (analyze.py).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 6.5, "axes.linewidth": 0.6})
INK2 = "#52514e"
DESIGNS = [("v1_n100", "① 여유 1.5mm"), ("n100", "② 여유 9mm"), ("v4_n100", "③ +같은 벽"), ("v5_n100", "④ +물러나기")]
CATS = [  # label, modes, colour
    ("성공", ("success",), "#c9e7d8"),
    ("못 집음", ("not_placed", "no_grasp"), "#eb6834"),
    ("기울어짐", ("tilted",), "#eda100"),
    ("어긋남", ("misplaced", "wrong_height"), "#2a78d6"),
    ("앞 통 밀침", ("pushed_previous",), "#7a5cc4"),
]


def main(out: Path) -> None:
    s = json.loads((ROOT / "results" / "summary.json").read_text())
    rows = [(lab, s[k]) for k, lab in DESIGNS if k in s]
    if len(rows) < 2:
        print("not enough results")
        return
    fig, ax = plt.subplots(figsize=(3.3, 0.28 * len(rows) + 0.55), dpi=300)
    for i, (lab, v) in enumerate(rows):
        counts = {}
        for st in (1, 2, 3):
            for mode, n in v[f"stage{st}"]["modes"].items():
                counts[mode] = counts.get(mode, 0) + n
        total = sum(counts.values())
        left = 0.0
        for cname, modes, col in CATS:
            w = 100 * sum(counts.get(m, 0) for m in modes) / total
            if w > 0:
                ax.barh(i, w, left=left, color=col, height=0.62, edgecolor="white", linewidth=0.5,
                        label=cname if i == 0 or cname not in [h.get_label() for h in ax.patches] else None)
                if w >= 4 and cname != "성공":
                    ax.text(left + w / 2, i, f"{w:.0f}", ha="center", va="center", fontsize=5.6, color="white")
                left += w
        ax.text(101, i, f"{100 * counts.get('success', 0) / total:.0f}%", va="center", fontsize=6, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xlabel("단계별 평가 150회 중 비율 (%)", color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in CATS]
    ax.legend(handles, [c[0] for c in CATS], frameon=False, fontsize=5.8, ncol=5, loc="lower center",
              bbox_to_anchor=(0.45, 1.0), handlelength=1.0, columnspacing=0.8, handletextpad=0.3)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0, colors=INK2)
    ax.tick_params(axis="x", colors=INK2, length=2)
    fig.tight_layout(pad=0.3)
    fig.savefig(out, dpi=300)
    print("saved", out)


if __name__ == "__main__":
    main(ROOT / "paper" / "fig_failures.png")
