"""Training loss of the three stage policies (one condition), from runs/s*_<name>/train_log.json."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT = Path(__file__).resolve().parents[1]
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 7, "axes.linewidth": 0.6})
name = sys.argv[1] if len(sys.argv) > 1 else "n100"
fig, ax = plt.subplots(figsize=(3.3, 1.9), dpi=300)
for s, (lab, col, ls) in enumerate((("1층", "#2a78d6", "-"), ("옆", "#eb6834", "--"), ("2층", "#1baf7a", ":")), 1):
    log = json.loads((ROOT / "runs" / f"s{s}_{name}" / "train_log.json").read_text())
    ax.plot([r["step"] for r in log], [r["loss"] for r in log], color=col, lw=1.0, ls=ls, label=lab)
ax.set_yscale("log")
ax.set_xlabel("학습 스텝", color="#52514e")
ax.set_ylabel("손실 (L1 + 10·KL)", color="#52514e")
ax.legend(frameon=False, fontsize=6.5)
ax.grid(axis="y", color="#e4e3df", lw=0.5, which="both")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout(pad=0.3)
out = ROOT / "paper" / f"fig_loss_{name}.png"
fig.savefig(out, dpi=300)
print("saved", out)
