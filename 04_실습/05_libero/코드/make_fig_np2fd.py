"""10/23 일지 그림: 새 기준 28항목, 예전 보조(piper_np2mc) vs 고친 보조(piper_np2fd), 같은 장면 20개씩.
   python make_fig_np2fd.py → outputs/media/np2fd_compare.png"""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

for f in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"]:
    font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"
OLD, NEW = "#b8b8b8", "#2e6fbf"
NAME = {0: "파랑→흰", 1: "빨강→주황", 3: "빨강→파랑 위", 4: "빨강→회색", 5: "초록→흰", 7: "노랑 막대→흰", 8: "빨강→보라", 9: "초록→파랑 위"}


def load(d, a, b):
    out = {}
    for f in glob.glob(f"{d}/A{a}_B{b}_*.json"):
        for e in json.load(open(f))["episodes"]:
            out[e["episode"]] = e
    return out


RES = [(8, 5), (8, 7), (8, 9), (4, 5), (4, 9), (1, 7)]
PAIRS = [(8, 0), (8, 3), (8, 5), (8, 7), (8, 9), (4, 5), (4, 9), (1, 7), (8, 1), (8, 4), (1, 8), (4, 1)]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1, 1.6]})
w = 0.38


def bars(axis, labels, old, new, title):
    xs = np.arange(len(labels))
    for i, (vals, col, lab) in enumerate(((old, OLD, "예전 보조"), (new, NEW, "고친 보조"))):
        r = axis.bar(xs + (i - 0.5) * (w + 0.02), vals, w, color=col, label=lab)
        for rr, v in zip(r, vals):
            axis.text(rr.get_x() + rr.get_width() / 2, v + 0.25, str(v), ha="center", va="bottom", fontsize=9, color=INK)
    axis.set_xticks(xs, labels, fontsize=9)
    axis.set_ylim(0, 21.5)
    axis.set_yticks([0, 5, 10, 15, 20])
    axis.axhline(18, color="#d35400", lw=1, ls="--")
    axis.text(-0.62, 18, "90%", color="#d35400", fontsize=9, ha="right", va="center")
    axis.set_ylabel("성공 (20장면 중)", color=MUTED)
    axis.grid(axis="y", color=GRID, lw=0.8)
    axis.set_axisbelow(True)
    axis.set_title(title, loc="left", fontsize=13, color=INK)
    for s in ("top", "right"):
        axis.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axis.spines[s].set_color(GRID)


ro, rn, lab = [], [], []
for a, b in RES:
    o, n = load("outputs/piper_np2mc_switch", a, b), load("outputs/piper_np2fd_switch", a, b)
    ro.append(sum(bool(e.get("switched") and e.get("a_resume_genuine")) for e in o.values()))
    rn.append(sum(bool(e.get("switched") and e.get("a_resume_genuine")) for e in n.values()))
    lab.append(f"{a}→{b}→{a}")
bars(ax, lab, ro, rn, f"돌아오기 (6쌍) {100 * sum(ro) / 120:.1f}% → {100 * sum(rn) / 120:.1f}%")
fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=11, loc="upper right", ncol=2)
bo, bn, lab = [], [], []
for a, b in PAIRS:
    o, n = load("outputs/piper_np2mc_switch", a, b), load("outputs/piper_np2fd_switch", a, b)
    bo.append(sum(bool(e.get("switched") and e.get("b_success")) for e in o.values()))
    bn.append(sum(bool(e.get("switched") and e.get("b_success")) for e in n.values()))
    lab.append(f"{a}→{b}")
bars(bx, lab, bo, bn, f"일 바꾸기 B (12쌍) {100 * sum(bo) / 240:.1f}% → {100 * sum(bn) / 240:.1f}%")
fig.suptitle("새 기준(옮기는 도중 새 지시) np2 + 보조, 같은 장면끼리", x=0.01, ha="left", fontsize=14, color=INK)
fig.tight_layout()
fig.savefig("outputs/media/np2fd_compare.png", dpi=150)
print("outputs/media/np2fd_compare.png", sum(ro), sum(rn), sum(bo), sum(bn))
