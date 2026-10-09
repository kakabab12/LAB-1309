"""Journal figures for the pallet (10/9): v1 vs v2 layout (top view with the arm's reach band) and where the slot-7
bin ends up relative to its target (placement offsets from pallet_slot_check.py)."""

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Rectangle

font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 8})
W = 64  # bin width (mm)


def layout(ax, x0, y0, gap_y, title):
    X = lambda x, y: (-y, x)  # robot frame (x forward, y left) -> plot (right, up)
    for r_mm, ls in ((190, ":"), (290, "--")):
        ax.add_patch(Circle((0, 0), r_mm, fill=False, ls=ls, lw=0.8, ec="#888"))
    ax.text(*X(290 * math.cos(0.75), 290 * math.sin(0.75)), "팔 닿는 끝 29cm", fontsize=7, color="#666", ha="right")
    ax.add_patch(Rectangle((-55, -55), 110, 110, fc="#d9d6cf", ec="#555"))
    ax.text(0, 0, "로봇", ha="center", va="center", fontsize=7)
    px, py = X(205, -140)
    ax.add_patch(Rectangle((px - 62, py - 70), 124, 140, fc="#55565b", ec="k"))
    ax.text(px, py, "저울", color="white", ha="center", va="center")
    labels = {(1, 1): "1·5", (0, 1): "2·6", (1, 0): "3·7", (0, 0): "4·8"}
    for r in (0, 1):
        for c in (0, 1):
            x, y = x0 + r * (W + 8), y0 + c * (W + gap_y)
            cx, cy = X(x, y)
            far_corner = r == 1 and c == 1
            ax.add_patch(Rectangle((cx - W / 2, cy - W / 2), W, W, fc="#3b6fd6" if not (r == 1 and c == 0) else "#e0703a",
                                   ec="k", alpha=0.9))
            ax.text(cx, cy, labels[(r, c)], color="white", ha="center", va="center", fontsize=8, weight="bold")
            if far_corner:
                ax.text(cx - 4, cy + W / 2 + 6, f"{math.hypot(x, y) / 10:.1f}cm", fontsize=7, ha="center")
    gx, gy = X(x0 + W + 8 + W / 2 + 6, y0 + W / 2 + gap_y / 2)
    ax.text(gx, gy, f"열 간격\n{gap_y:.0f}mm", ha="center", va="bottom", fontsize=7, color="#c0392b")
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-230, 215); ax.set_ylim(-80, 330); ax.set_aspect("equal"); ax.axis("off")


fig, axs = plt.subplots(1, 2, figsize=(8, 3.6), dpi=150)
layout(axs[0], 200, 30, 8, "v1 (x0 20cm, y0 3.0cm, 열 간격 8mm)")
layout(axs[1], 200, 17, 14, "v2 (x0 20cm, y0 1.7cm, 열 간격 14mm)")
fig.suptitle("팔레트 배치 (위에서 본 모습, 숫자 = 1층·2층 칸 번호, 주황 = 7번 칸이 쌓이는 자리)", fontsize=9)
fig.tight_layout()
fig.savefig("results/pallet_layout_v1_v2.png")

fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=150)
for f, lab, col, mk in (("results/pallet/slotcheck_s6_n0.003.json", "100스텝 실행 (시연 100)", "#9a9994", "o"),
                        ("results/pallet/slotcheck_s6_k25.json", "25스텝 재예측 (시연 100)", "#2a78d6", "s"),
                        ("results/pallet/slotcheck_s6b_k25.json", "25스텝 재예측 (시연 300)", "#eb6834", "^")):
    rows = json.load(open(f))["rows"]
    ok = [r for r in rows if r["success"]]
    bad = [r for r in rows if not r["success"] and abs(r["dxy_mm"][0]) < 40 and abs(r["dxy_mm"][1]) < 40]
    ax.scatter([r["dxy_mm"][1] for r in ok], [r["dxy_mm"][0] for r in ok], s=16, color=col, marker=mk,
               label=f"{lab}: {len(ok)}/{len(rows)}")
    ax.scatter([r["dxy_mm"][1] for r in bad], [r["dxy_mm"][0] for r in bad], s=30, facecolor="none", edgecolor=col, marker=mk)
ax.add_patch(Circle((0, 0), 15, fill=False, ls="--", ec="#c0392b"))
ax.axvline(8, color="#555", lw=1)
ax.text(8.5, -18, "v1: 옆 열 통까지\n간격 8mm", fontsize=7, color="#555")
ax.axvline(14, color="#2a9d55", lw=1, ls="-.")
ax.text(14.5, 12, "v2: 14mm", fontsize=7, color="#2a9d55")
ax.text(-14, 16, "허용 오차 15mm", fontsize=7, color="#c0392b")
ax.set_xlabel("옆 열 쪽 치우침 (mm, + = 이미 쌓인 옆 열 방향)")
ax.set_ylabel("앞뒤 치우침 (mm)")
ax.set_xlim(-22, 26); ax.set_ylim(-22, 22); ax.set_aspect("equal")
ax.legend(fontsize=6.5, loc="lower left", frameon=False)
ax.set_title("7번 칸 통이 놓인 위치 (빈 표시 = 실패, 범위 안만)", fontsize=8.5)
fig.tight_layout()
fig.savefig("results/pallet_slot7_offsets.png")
print("saved results/pallet_layout_v1_v2.png results/pallet_slot7_offsets.png")
