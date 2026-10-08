"""Journal figure: where the side bin (stage 2) lands along the row, design ④ (v5_n100) vs ⑤ (v8_n100).

Per-stage extended evaluation (200 trials each, results/eval/*_x200). Offset = bin y - slot y (stack_env.T2);
the success tolerance is 15 mm in xy, so bins beyond +15 mm fail although they stand upright.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from stack_env import T2

font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 9})
RUNS = [("v5_n100_x200", "④ 고정 칸 (최종)", "#2a78d6"), ("v8_n100_x200", "⑤ 1층 통 따라 이동", "#eb6834")]

fig, ax = plt.subplots(figsize=(6, 2.8), dpi=150)
bins = [x - 0.5 for x in range(-10, 24)]
for name, label, col in RUNS:
    tr = json.loads(Path(f"results/eval/{name}/results.json").read_text())["per_stage"]["2"]["trials"]
    dy = [1000 * (t["bin_pos"][1] - T2[1]) for t in tr]
    fails = sum(not t["success"] for t in tr)
    ax.hist(dy, bins=bins, color=col, alpha=0.55, label=f"{label}: 실패 {fails}/{len(tr)}")
ax.axvline(15, color="#c0392b", lw=1.2, ls="--")
ax.text(15.3, ax.get_ylim()[1] * 0.9, "허용 오차 15mm", color="#c0392b", fontsize=8)
ax.set_xlabel("옆 통이 놓인 위치 − 목표 칸 (mm, + = 1층 통에서 멀어지는 쪽)")
ax.set_ylabel("시행 수")
ax.legend(frameon=False, fontsize=8, loc="upper left")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig("results/side_offset_v5_v8.png")
print("saved results/side_offset_v5_v8.png")
