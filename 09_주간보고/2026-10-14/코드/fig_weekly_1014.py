"""Picture strip for the weekly report of 10-14: simulation scene | success by demonstration design | wrist-angle gap."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "act_sim" / "paper"
OUT = HERE / "weekly_1014_strip.png"

# Three-stage (chained) success, simulation, 50 scenes each (paper Table 1); last bar = 200 scenes
labels = ["Initial", "+ Grasp\nclearance", "+ Same\ngrasp wall", "+ Retreat\nafter\nrelease", "All rules\n(200\nscenes)"]
vals = [28, 62, 82, 96, 99.0]
cols = ["#9aa5b1", "#7aa6d6", "#4b86c8", "#1c64c8", "#0b3d91"]
fig, ax = plt.subplots(figsize=(4.0, 2.6), dpi=250)
bars = ax.bar(range(len(vals)), vals, color=cols, width=0.68)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%" if v % 1 == 0 and v > 98 else f"{v:g}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_xticks(range(len(vals)), labels, fontsize=6.3)
ax.set_ylim(0, 118)
ax.set_yticks([0, 50, 100])
ax.tick_params(axis="y", labelsize=7)
ax.set_ylabel("3-stage success (%)", fontsize=7.5)
ax.set_title("Demonstration rules (100 demos per stage, sim)", fontsize=8)
ax.text(-0.35, 104, "Retry with scale weight: 200/200", ha="left", fontsize=6.8, color="#0b3d91", fontweight="bold")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
chart = HERE / "_chart.png"
fig.savefig(chart)
plt.close(fig)

H = 900
ims = [Image.open(PAPER / "prof" / "fig_sim_top_en.png").convert("RGB"), Image.open(chart).convert("RGB")]
ims = [im.resize((round(im.width * H / im.height), H), Image.LANCZOS) for im in ims]
gap = 40
strip = Image.new("RGB", (sum(im.width for im in ims) + gap * (len(ims) - 1), H), "white")
x = 0
for im in ims:
    strip.paste(im, (x, 0))
    x += im.width + gap
strip.save(OUT)
chart.unlink()
print(OUT, strip.size)
