"""Poster figures drawn at print size (fonts >= 18 pt, so labels read at 1-2 m like the poster text):
  img/audit.png        pre-training check: wrist angle at grasp, human-like vs rule-based (stage 1, 100 demos)
  img/rule1.png ...    schematics of the three demonstration rules (before -> after)
usage: .venv/bin/python paper/poster/fig_poster.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, patches
from matplotlib import transforms as mtransforms

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "img"
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 20, "axes.linewidth": 1.2})
HUMAN, RULE, NAVY, OK, BAD = "#d9480f", "#1c5fc8", "#14213d", "#2b8a3e", "#c92a2a"
MM = 1 / 25.4


def audit():
    rows = json.loads((ROOT / "results" / "demo_audit_general.json").read_text())
    objs = [("bin", "정사각형 적재함"), ("box", "직사각형 상자"), ("cup", "손잡이 컵")]
    sets = [(True, "사람식 시연", HUMAN), (False, "규칙 시연", RULE)]
    fig, axs = plt.subplots(3, 1, figsize=(400 * MM, 215 * MM), dpi=250, sharex=True)
    for ax, (obj, name) in zip(axs, objs):
        for k, (naive, label, col) in enumerate(sets):
            v = next(v for v in rows.values()
                     if v["object"] == obj and v["layout"] == "orig" and v["naive"] == naive and v["stage"] == 1)
            y = 1 - k
            roll = np.sort(np.asarray(v["roll"]))
            ax.vlines(roll, y - 0.34, y + 0.34, color=col, lw=1.6, alpha=0.8)
            d = np.diff(roll); i = int(np.argmax(d)); lo, hi = roll[i], roll[i + 1]
            if hi - lo >= 10:
                ax.axvspan(lo, hi, ymin=(y + 0.5 - 0.44) / 2, ymax=(y + 0.5 + 0.44) / 2, color=col, alpha=0.13, lw=0)
                ax.text((lo + hi) / 2, y, f"빈 구간 {v['widest_gap_deg']:.0f}°", ha="center", va="center",
                        fontsize=19, color=col, fontweight="bold")
            else:
                ax.text(roll.max() + 5, y, f"빈 구간 {v['widest_gap_deg']:.0f}°", ha="left", va="center",
                        fontsize=19, color=col, fontweight="bold")
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([1, 0], [s[1] for s in sets], fontsize=19)
        for t, (_, _, col) in zip(ax.get_yticklabels(), sets):
            t.set_color(col); t.set_fontweight("bold")
        ax.set_title(name, fontsize=21, loc="left", pad=4, fontweight="bold", color=NAVY)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", lw=0.8, alpha=0.35)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    axs[-1].set_xlim(-110, 125)
    axs[-1].set_xticks(range(-100, 125, 50))
    axs[-1].tick_params(axis="x", labelsize=18)
    axs[-1].set_xlabel("그리퍼가 닫히는 순간의 손목 회전각 (°)", fontsize=19)
    fig.tight_layout(h_pad=0.4)
    fig.savefig(OUT / "audit.png")
    plt.close(fig)


# ------------------------------------------------------------------ rule schematics (mm coordinates)
# One image per rule: left half = before (red), right half = after (green). Only numbers are drawn here; the
# before / after titles and the explanation are poster text (native, editable).
WALL_C, BIN_F, JAW_C, JAW2_C = "#1f3f8f", "#c9d7f5", "#2d2d2d", "#8a8a8a"


def setup(ax, xl, yl):
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.axis("off")


def badge(ax, x, y, good, r=6.5):
    col = OK if good else BAD
    ax.add_patch(patches.Circle((x, y), r, fc=col, ec="none", zorder=10))
    if good:
        ax.plot([x - 3.2, x - 0.8, x + 3.6], [y + 0.2, y - 2.6, y + 3.0], color="white", lw=4.2,
                solid_capstyle="round", solid_joinstyle="round", zorder=11)
    else:
        for dx in (1, -1):
            ax.plot([x - 3 * dx, x + 3 * dx], [y - 3, y + 3], color="white", lw=4.2, solid_capstyle="round", zorder=11)


def rect(ax, x, y, w, h, fc, ec="none", lw=0, z=3, t=None, alpha=1.0):
    kw = dict(fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha)
    if t is not None:
        kw["transform"] = t
    ax.add_patch(patches.Rectangle((x, y), w, h, **kw))


def arrow(ax, xy, xytext, col, style="->", lw=3, ms=22):
    ax.annotate("", xy=xy, xytext=xytext, arrowprops=dict(arrowstyle=style, lw=lw, color=col, shrinkA=0, shrinkB=0,
                                                          mutation_scale=ms), zorder=8)


def canvas(xl, yl, width_mm):
    """One axes, equal aspect, figure sized to the data box at print size (so 20 pt labels print at 7 mm)."""
    w = width_mm
    h = w * (yl[1] - yl[0]) / (xl[1] - xl[0])
    fig = plt.figure(figsize=(w * MM, h * MM), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    setup(ax, xl, yl)
    return fig, ax


def sep(ax, x, y0, y1):
    ax.plot([x, x], [y0, y1], color="#c9d3e3", lw=1.5, zorder=1)


def dashed(ax, x, y, w, h):
    rect(ax, x, y, w, h, "none", ec="#9a9a9a", lw=2, z=4)
    ax.patches[-1].set_linestyle((0, (4, 3)))


FS = 20  # label size (pt) = 7 mm in print


def rule1():
    fig, ax = canvas((-14, 77), (-10, 64), 104)
    for X in (0, 50):
        rect(ax, X + 4, 0, 13, 2.6, BIN_F, z=2)           # bin floor
        rect(ax, X, 0, 4, 30, WALL_C)                      # bin wall (section)
    sep(ax, 33, -8, 56)
    # before: lowered flush with the wall; 3 mm toward it and the fixed jaw sits on the wall top
    dashed(ax, -7, 8, 7, 18)
    rect(ax, -3, 30, 7, 18, JAW_C, z=5)
    arrow(ax, (1.5, 52), (-6, 52), BAD, lw=2.6, ms=16)
    ax.text(-2, 53.5, "3 mm", fontsize=FS, ha="center", va="bottom", color=BAD, fontweight="bold")
    badge(ax, 18, 42, False, r=4.6)
    # after: lowered 9 mm outside the wall, the wall fits anywhere in the 0-17 mm opening
    rect(ax, 34, 8, 7, 18, JAW_C, z=5)                    # fixed jaw, 9 mm outside
    rect(ax, 60, 8, 7, 18, JAW2_C, z=5)                   # moving jaw (open)
    rect(ax, 41, 4.2, 19, 2.8, OK, z=1, alpha=0.45)
    arrow(ax, (50, 29), (41, 29), NAVY, style="<->", lw=2.2, ms=12)
    ax.text(45.5, 31, "9 mm", fontsize=FS, ha="center", va="bottom", color=NAVY, fontweight="bold")
    ax.text(50.5, -1.5, "0~17 mm", fontsize=FS, ha="center", va="top", color=OK, fontweight="bold")
    badge(ax, 68, 42, True, r=4.6)
    fig.savefig(OUT / "rule1.png", facecolor="white")
    plt.close(fig)


def square(ax, cx, cy, deg, wall, col, s=20):
    t = mtransforms.Affine2D().rotate_deg_around(cx, cy, deg) + ax.transData
    rect(ax, cx - s / 2, cy - s / 2, s, s, BIN_F, ec=WALL_C, lw=2.2, t=t, z=3)
    if wall == "front":   # wall facing the robot (bottom); gripper bar just outside it
        seg, jaw_xy, jw, jh = ((cx - s / 2, cy - s / 2), (cx + s / 2, cy - s / 2)), (cx - 4.5, cy - s / 2 - 5), 9, 3
    else:                 # side wall (right)
        seg, jaw_xy, jw, jh = ((cx + s / 2, cy - s / 2), (cx + s / 2, cy + s / 2)), (cx + s / 2 + 2, cy - 4.5), 3, 9
    (x0, y0), (x1, y1) = seg
    ax.plot([x0, x1], [y0, y1], color=col, lw=7, solid_capstyle="round", transform=t, zorder=4)
    rect(ax, *jaw_xy, jw, jh, JAW_C, t=t, z=5)
    ax.text(cx, cy + s / 2 + 4, f"+{deg:.0f}°", fontsize=FS, ha="center", va="bottom", color=NAVY, fontweight="bold")


def rule2():
    fig, ax = canvas((-14, 122), (-28, 30), 104)
    square(ax, 2, 6, 7, "front", BAD); square(ax, 32, 6, 9, "side", BAD)
    sep(ax, 54, -26, 28)
    square(ax, 76, 6, 7, "front", OK); square(ax, 106, 6, 9, "front", OK)
    badge(ax, 17, -20, False, r=5.2); badge(ax, 91, -20, True, r=5.2)
    fig.savefig(OUT / "rule2.png", facecolor="white")
    plt.close(fig)


def rule3():
    fig, ax = canvas((-16, 121), (-3, 76), 104)
    for X in (0, 78):
        rect(ax, X, 0, 40, 26, BIN_F, ec=WALL_C, lw=2.2)                         # 1st floor
    sep(ax, 50, -2, 74)
    # before: lifting right after release drags the upper bin up by its outer wall -> it tilts
    t = mtransforms.Affine2D().rotate_deg_around(40, 26, -12) + ax.transData
    rect(ax, 0, 26, 40, 26, BIN_F, ec=BAD, lw=2.2, t=t, z=4)
    rect(ax, -7, 40, 5, 15, JAW_C, z=5)
    arrow(ax, (-4.5, 71), (-4.5, 57), BAD, lw=2.6, ms=16)
    badge(ax, 33, 66, False, r=4.6)
    # after: open, move 5 mm outward, then lift
    rect(ax, 78, 26, 40, 26, BIN_F, ec=WALL_C, lw=2.2, z=4)
    dashed(ax, 72, 34, 5, 15)
    rect(ax, 66, 34, 5, 15, JAW_C, z=5)
    arrow(ax, (65.5, 31), (72, 31), OK, lw=2.4, ms=14)
    ax.text(64.5, 28.5, "5 mm", fontsize=18, ha="center", va="top", color=OK, fontweight="bold")
    arrow(ax, (68.5, 71), (68.5, 51), OK, lw=2.6, ms=16)
    badge(ax, 111, 66, True, r=4.6)
    fig.savefig(OUT / "rule3.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    audit(); rule1(); rule2(); rule3()
    print("figures written to", OUT)
