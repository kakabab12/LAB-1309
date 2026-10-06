"""Project overview: real system vs. simulation study (for the repo / explanation, not the paper)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = "Noto Sans CJK JP"
INK, EDGE = "#1f2933", "#52606d"
C = {"real": "#e8f0fb", "sim": "#e6f4ea", "same": "#fdf1dc", "out": "#f1f1f1", "lost": "#fbe4e4"}


def box(ax, x, y, w, h, title, sub, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.015", fc=C[kind], ec=EDGE, lw=0.9))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=9, weight="bold", color=INK)
    ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=7, color=INK, linespacing=1.3)


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=1.0, color=EDGE))


fig = plt.figure(figsize=(10, 4.6), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.46)
ax.axis("off")

ax.text(0.01, 0.44, "실제 시스템 (캡스톤, 2026 상반기)", fontsize=10, weight="bold", color="#2a5aa0", va="top")
ax.text(0.01, 0.215, "시뮬레이션 평가 (이번 논문)", fontsize=10, weight="bold", color="#2f7d4a", va="top")

w, h, y1, y2 = 0.145, 0.13, 0.27, 0.045
xs = [0.01, 0.175, 0.34, 0.505, 0.67, 0.835]
real = [("무게 판정", "컨베이어 색 분류\nYOLOv8 7-segment\n118g 이상 → MCP", "real"),
        ("시연자", "사람이 리더암 조작\n(원격조작)", "real"),
        ("시연 데이터", "단계별 100 · 200회\n손목·상단 영상+관절", "real"),
        ("ACT 학습", "LeRobot 0.3.3\nPC (L4 / RTX 3060)", "same"),
        ("실행", "Jetson Orin Nano\nSO-101 실물", "real"),
        ("결과", "동작 확인\n수치 기록 유실·로봇 파손", "lost")]
sim = [("무게 판정", "실측 수치 그대로 사용\n(FPS 59.5, 91.7% 등)", "out"),
       ("시연자", "스크립트 전문가\n(통 위치 알고 역기구학)", "sim"),
       ("시연 데이터", "단계별 100·200·500·1000\n+ DART 잡음 200", "sim"),
       ("ACT 학습", "같은 ACT 설정\nGTX 1080 Ti", "same"),
       ("평가", "ACT만 제어 (스크립트 X)\n단계별·연속 각 50회", "sim"),
       ("논문", "성공률·오차 누적\n시연 수 비교, DART", "out")]
for i, (t, s, k) in enumerate(real):
    box(ax, xs[i], y1, w, h, t, s, k)
    if i:
        arrow(ax, (xs[i - 1] + w, y1 + h / 2), (xs[i], y1 + h / 2))
for i, (t, s, k) in enumerate(sim):
    box(ax, xs[i], y2, w, h, t, s, k)
    if i:
        arrow(ax, (xs[i - 1] + w, y2 + h / 2), (xs[i], y2 + h / 2))
for i in (1, 3, 4):
    ax.annotate("", xy=(xs[i] + w / 2, y2 + h + 0.004), xytext=(xs[i] + w / 2, y1 - 0.004),
                arrowprops=dict(arrowstyle="-|>", ls="--", color="#9a9994", lw=0.9))
ax.text(xs[1] + w / 2 + 0.006, (y1 + y2 + h) / 2, "사람 대신", fontsize=7, color="#52514e", va="center")
ax.text(xs[3] + w / 2 + 0.006, (y1 + y2 + h) / 2, "동일", fontsize=7, color="#52514e", va="center")
ax.text(xs[4] + w / 2 + 0.006, (y1 + y2 + h) / 2, "같은 로봇·카메라\n30Hz 재현", fontsize=7, color="#52514e", va="center")
fig.savefig("paper/fig_overview.png", dpi=200)
print("saved paper/fig_overview.png")
