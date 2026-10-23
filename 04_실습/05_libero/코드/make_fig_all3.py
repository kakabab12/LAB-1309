"""10/23 오후 일지 그림: '셋 다' (B 성공 + 돌아오기 + B 유지) 실패를 종류별로, 설정마다 (새 기준 돌아오기 6쌍 120장면).
   python make_fig_all3.py → outputs/media/all3_breakdown.png"""
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
RES = ["8:5", "8:7", "8:9", "4:5", "4:9", "1:7"]
KINDS = [("B실패", "B 를 못 함", "#c0392b"), ("돌아오기실패", "A 로 못 돌아옴", "#d35400"), ("B무너짐", "쌓은 B 무너짐", "#8e44ad"),
         ("전환X", "A 를 먼저 끝냄", "#9a9a9a")]


def count(dirs):
    c = {k: 0 for k, *_ in KINDS}
    c["성공"] = 0
    for p in RES:
        a, b = p.split(":")
        es = {}
        for d in dirs.get(p, dirs["*"]):
            for f in glob.glob(f"{d}_switch/A{a}_B{b}_*.json"):
                for e in json.load(open(f))["episodes"]:
                    es[e["episode"]] = e
        for e in es.values():
            if not e.get("switched"):
                k = "전환X"
            elif not e.get("b_success"):
                k = "B실패"
            elif not e.get("a_resume_genuine"):
                k = "돌아오기실패"
            elif not e.get("b_still_satisfied_at_end"):
                k = "B무너짐"
            else:
                k = "성공"
            c[k] += 1
    return c


CFG = [("np2 + 예전 보조", {"*": ["outputs/piper_np2mc"]}),
       ("np2 + 고친 보조", {"*": ["outputs/piper_np2fd"]}),
       ("np3 + 고친 보조", {"*": ["outputs/piper_np3fd"]}),
       ("np3 + 새 설정", {"*": ["outputs/piper_np3x"]}),
       ("np3 + 새 설정\n+ 쌓은 블록 지키기", {"*": ["outputs/piper_np3x"], "8:9": ["outputs/piper_np3h"], "4:9": ["outputs/piper_np3h"]})]
rows = [(n, count(d)) for n, d in CFG]
fig, ax = plt.subplots(figsize=(11, 4.8))
ys = np.arange(len(rows))[::-1]
for y, (n, c) in zip(ys, rows):
    left = 0
    for k, lab, col in KINDS:
        v = c[k]
        if v:
            ax.barh(y, v, left=left, color=col, height=0.6, edgecolor="white", linewidth=1.5)
            if v >= 2:
                ax.text(left + v / 2, y, str(v), ha="center", va="center", color="white", fontsize=10)
        left += v
    ax.text(left + 0.4, y, f"셋 다 성공 {c['성공']}/120 ({100 * c['성공'] / 120:.1f}%)", va="center", fontsize=10, color=INK)
ax.set_yticks(ys, [n for n, _ in rows], fontsize=10)
ax.set_xlim(0, 52)
ax.set_ylim(ys[-1] - 0.8, ys[0] + 0.5)
ax.set_xlabel("실패한 장면 수 (120장면 중)", color=MUTED)
ax.axvline(12, color="#2e6fbf", lw=1, ls="--")
ax.text(12.3, ys[-1] - 0.45, "90% 선 (실패 12개)", color="#2e6fbf", fontsize=9, va="top")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=col, label=lab) for k, lab, col in KINDS], frameon=False, fontsize=10, loc="lower right", ncol=2)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.set_title("새 기준 '셋 다' 실패를 종류별로 (돌아오기 6쌍, 같은 장면 120개)", loc="left", fontsize=13, color=INK)
fig.tight_layout()
fig.savefig("outputs/media/all3_breakdown.png", dpi=150)
print("outputs/media/all3_breakdown.png", [(n.replace(chr(10), " "), c["성공"]) for n, c in rows])
