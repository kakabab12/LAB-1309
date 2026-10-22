"""10/22 밤 일지 그림: 판이 밀린 순간 (diag_pad, A4→B9 장면 1000) + 실패 20장면 고치기 전/후.
   python make_fig_padpush.py → outputs/media/padpush_fix.png"""
import glob
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np

for f in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"]:
    font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"

# 진단 기록에서 A4→B9 1000 의 판·블록 움직임
log = open("outputs/piper/diag_pad.log").read()
sec = log.split("=== A4 B9 장면 1000")[1].split("===")[0]
red = [(int(m[0]), float(m[1]), float(m[2])) for m in re.findall(r"\[A2 (\d+)\] 판 움직임 [^|]+\|[^|]+\| 블록 \[([-\d.]+), ([-\d.]+)", sec)]
t = np.load("outputs/piper_diag_pad/traj/A4_B9_grasp30_keep_lat9_ep1000.npz")
hand = t["pos"][505:536]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})
# 왼쪽: 위에서 본 책상 (가로 = 좌우 y, 세로 = 앞뒤 x)
P0, P1 = (0.010, -0.126), (0.001, -0.063)
ax.add_patch(Rectangle((P0[1] - 0.05, P0[0] - 0.05), 0.10, 0.10, fill=False, ls="--", lw=1.6, ec="#8a8a8a"))
ax.add_patch(Rectangle((P1[1] - 0.05, P1[0] - 0.05), 0.10, 0.10, fc="#9a9a9a", alpha=0.35, ec="#5a5a5a", lw=1.6))
ax.text(P0[1], P0[0] + 0.054, "회색 판 처음 자리", ha="center", va="bottom", color=MUTED, fontsize=10)
ax.text(P1[1], P1[0] - 0.054, "밀린 뒤 (오른쪽으로 6.3cm)", ha="center", va="top", color=INK, fontsize=10)
ax.plot(hand[:, 1], hand[:, 0], color=INK, lw=2, label="손 (책상 높이 3cm 안)")
ry, rx = [r[2] for r in red], [r[1] for r in red]
ax.plot([-0.218] + ry, [0.008] + rx, color="#c0392b", lw=2, marker="s", ms=5, label="빨강 블록 (끌려감)")
ax.plot(-0.212, 0.039, marker="x", color="#d35400", ms=11, mew=2.5, ls="none", label="보조가 잰 빨강 (3.2cm 틀림)")
ax.annotate("다시 쥔 자리", (-0.218, 0.008), (-0.255, -0.035), color=INK, fontsize=10, arrowprops=dict(arrowstyle="-", color=MUTED))
ax.set_xlim(-0.30, 0.0)
ax.set_ylim(-0.12, 0.09)
ax.set_aspect("equal")
ax.set_xlabel("책상 좌우 (m)", color=MUTED)
ax.set_ylabel("책상 앞뒤 (m)", color=MUTED)
ax.grid(color=GRID, lw=0.8)
ax.legend(loc="lower left", frameon=False, fontsize=10)
ax.set_title("돌아오기에서 판이 밀린 순간 (A4→B9 장면 1000)", loc="left", fontsize=13, color=INK)
for s in ax.spines.values():
    s.set_color(GRID)

# 오른쪽: 실패 20장면 고치기 전/후 (같은 장면 번호)
base = {}
for d in ["outputs/piper_release_test", "outputs/piper_np2mc_switch"]:
    for f in glob.glob(d + "/A*.json"):
        for e in json.load(open(f))["episodes"]:
            base[(f.split("/")[-1][:5], e["episode"])] = e
cnt = {"전": [0, 0], "후": [0, 0]}
for f in glob.glob("outputs/piper_fix_test/e*/A*.json"):
    for e in json.load(open(f))["episodes"]:
        b = base[(f.split("/")[-1][:5], e["episode"])]
        for k, x in (("전", b), ("후", e)):
            cnt[k][0] += bool(x.get("a_resume_genuine"))
            cnt[k][1] += bool(x.get("b_success") and x.get("a_resume_genuine") and x.get("b_still_satisfied_at_end"))
labels = ["돌아오기 성공", "B·돌아오기·B 유지\n셋 다 성공"]
xs = np.arange(2)
w = 0.34
for i, (k, col) in enumerate((("전", "#b8b8b8"), ("후", "#2e6fbf"))):
    bars = bx.bar(xs + (i - 0.5) * (w + 0.02), cnt[k], w, color=col, label="고치기 전" if k == "전" else "고친 뒤")
    for r, v in zip(bars, cnt[k]):
        bx.text(r.get_x() + r.get_width() / 2, v + 0.3, f"{v}", ha="center", va="bottom", color=INK, fontsize=12)
bx.set_xticks(xs, labels, fontsize=11)
bx.set_ylim(0, 20)
bx.set_yticks([0, 5, 10, 15, 20])
bx.set_ylabel("장면 수 (20개 중)", color=MUTED)
bx.grid(axis="y", color=GRID, lw=0.8)
bx.set_axisbelow(True)
bx.legend(frameon=False, fontsize=10, loc="upper left")
bx.set_title("원래 실패하던 20장면을 같은 장면으로 다시", loc="left", fontsize=13, color=INK)
for s in ("top", "right"):
    bx.spines[s].set_visible(False)
for s in ("left", "bottom"):
    bx.spines[s].set_color(GRID)
fig.tight_layout()
fig.savefig("outputs/media/padpush_fix.png", dpi=150)
print("outputs/media/padpush_fix.png", cnt)
