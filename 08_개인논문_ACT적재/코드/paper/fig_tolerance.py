"""Grasp success vs grasp-position error along the wall normal, for the v1 and v2 demonstration designs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT = Path(__file__).resolve().parents[1]
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 7, "axes.linewidth": 0.6})
d = json.loads((ROOT / "results" / "grasp_tolerance_all.json").read_text())
S = [("v1 (여유 1.5mm, 열림 0.32)", "여유 1.5mm (v1)", "#eb6834", "--", "s"),
     ("v2b (여유 9mm, 열림 0.32)", "여유 9mm (v2)", "#2a78d6", "-", "o")]
fig, ax = plt.subplots(figsize=(3.3, 1.8), dpi=300)
ax.axvspan(-30, 0, color="#f3f2ee", zorder=0)
ax.text(-13.5, 8, "벽 쪽(안쪽)", fontsize=6, color="#52514e")
for key, lab, col, ls, mk in S:
    xs = sorted(int(k) for k in d[key]); ys = [100 * d[key][str(x)] for x in xs]
    ax.plot(xs, ys, color=col, ls=ls, marker=mk, ms=3.5, lw=1.3, mec="white", mew=0.5, label=lab, zorder=3)
ax.set_xlim(-15, 15); ax.set_ylim(-5, 105)
ax.set_xlabel("파지 위치 오차 (mm, 벽 법선 방향)", color="#52514e"); ax.set_ylabel("파지·적재 성공 (%)", color="#52514e")
ax.legend(frameon=False, fontsize=6.3, loc="lower right")
ax.grid(axis="y", color="#e4e3df", lw=0.5)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig(ROOT / "paper" / "fig_tolerance.png", dpi=300); print("saved")
