"""Fig. 3 (paper, one column): the pre-training demonstration check, drawn so it reads at column width.

One row per object (stage-1 demonstrations, 100 each, results/demo_audit_general.json): every demonstration is a
tick at its wrist-roll angle when the gripper closes, human-like demonstrations above, rule-based below; the widest
empty band of each set is shaded and labelled. Replaces the overlapping histograms of demo_audit_general.png.
-> paper/fig3_audit.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 8.5, "axes.linewidth": 0.6})

EN = "--en" in __import__("sys").argv  # English labels for the journal-format (blind review) version
OBJ = ([("bin", "Square bin"), ("box", "Rectangular box"), ("cup", "Cup with a handle")] if EN else
       [("bin", "정사각형 적재함"), ("box", "직사각형 상자"), ("cup", "손잡이 컵")])
SETS = ([(True, "Human-like", "#e8590c"), (False, "Rule-based", "#1c64c8")] if EN else
        [(True, "사람식 시연", "#e8590c"), (False, "규칙 시연", "#1c64c8")])
GAP = "gap" if EN else "빈 구간"


def gap_band(roll):
    """Widest empty band between neighbouring sorted angles (same measure as demo_audit.widest_gap)."""
    r = np.sort(np.asarray(roll))
    d = np.diff(r)
    i = int(np.argmax(d))
    return r[i], r[i + 1]


def main() -> None:
    rows = json.loads((ROOT / "results" / "demo_audit_general.json").read_text())
    fig, axs = plt.subplots(3, 1, figsize=(3.35, 3.05), dpi=300, sharex=True)
    for ax, (obj, name) in zip(axs, OBJ):
        for k, (naive, label, col) in enumerate(SETS):
            v = next(v for v in rows.values()
                     if v["object"] == obj and v["layout"] == "orig" and v["naive"] == naive and v["stage"] == 1)
            y = 1 - k  # human-like on top
            roll = np.asarray(v["roll"])
            ax.vlines(roll, y - 0.32, y + 0.32, color=col, lw=0.7, alpha=0.75)
            lo, hi = gap_band(roll)
            if hi - lo >= 10:
                ax.axvspan(lo, hi, ymin=(y + 0.5 - 0.42) / 2, ymax=(y + 0.5 + 0.42) / 2, color=col, alpha=0.12, lw=0)
                ax.text((lo + hi) / 2, y, f"{GAP}\n{v['widest_gap_deg']:.0f}°", ha="center", va="center",
                        fontsize=7.5, color=col, fontweight="bold", linespacing=0.95)
            else:
                ax.text(roll.max() + 6, y, f"{GAP} {v['widest_gap_deg']:.0f}°", ha="left", va="center",
                        fontsize=7.5, color=col, fontweight="bold")
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([1, 0], [s[1] for s in SETS], fontsize=7.5)
        for t, (_, _, col) in zip(ax.get_yticklabels(), SETS):
            t.set_color(col)
        ax.set_title(name, fontsize=8.5, loc="left", pad=2, fontweight="bold")
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", lw=0.4, alpha=0.4)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[-1].set_xlim(-110, 125)
    axs[-1].set_xticks(range(-100, 125, 50))
    axs[-1].set_xlabel("Wrist roll angle when the gripper closes (°)" if EN else "그리퍼가 닫히는 순간의 손목 회전각 (°)")
    fig.tight_layout(h_pad=0.6)
    out = ROOT / "paper" / ("fig3_audit_en.png" if EN else "fig3_audit.png")
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
