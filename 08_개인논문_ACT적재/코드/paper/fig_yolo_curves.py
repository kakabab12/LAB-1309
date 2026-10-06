"""YOLOv8 training curves recorded in the team's checkpoints (number.pt, box.pt): validation mAP per epoch."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 7, "axes.linewidth": 0.6})
INK2, GRID = "#52514e", "#e4e3df"
SERIES = [("metrics/mAP50(B)", "mAP50", "#2a78d6", "-", "o"), ("metrics/mAP50-95(B)", "mAP50-95", "#eb6834", "--", "s")]

rec = json.loads((ROOT / "results" / "yolo" / "yolo_train_records.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.7), dpi=300, sharey=True)
for ax, (key, title) in zip(axes, (("number", "7-segment 숫자 (YOLOv8n)"), ("box", "상자 분류 (YOLOv8n-seg)"))):
    tr = rec[key]["train_results"]
    ep = [int(e) for e in tr["epoch"]]
    ends = []
    for k, lab, col, ls, mk in SERIES:
        ys = [100 * v for v in tr[k]]
        ax.plot(ep, ys, color=col, lw=1.2, ls=ls, marker=mk, ms=2.2, markevery=10, zorder=3)
        ends.append([ys[-1], f"{lab} {ys[-1]:.1f}"])
    ends.sort()
    if ends[1][0] - ends[0][0] < 9:
        ends[0][0] = ends[1][0] - 9
    for y, txt in ends:
        ax.text(ep[-1] + 1.5, y, txt, fontsize=5.4, va="center", color="#0b0b0b")
    ax.set_title(title, fontsize=6.8, color="#0b0b0b", pad=3)
    ax.set_xlim(0, 78)
    ax.set_ylim(0, 105)
    ax.set_xlabel("에폭", color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#9a9994")
    ax.tick_params(colors=INK2, length=2, width=0.5)
axes[0].set_ylabel("검증 정확도 (%)", color=INK2)
fig.tight_layout(pad=0.3, w_pad=0.6)
fig.savefig(ROOT / "paper" / "fig_yolo_curves.png", dpi=300)
print("saved paper/fig_yolo_curves.png")
