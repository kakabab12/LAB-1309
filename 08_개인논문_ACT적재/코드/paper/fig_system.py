"""Figure 1: overall system diagram (single-column width)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = "Noto Sans CJK JP"

INK = "#1f2933"
EDGE = "#52606d"
FILL = {"plant": "#e8f0fb", "edge": "#fdf1dc", "robot": "#e6f4ea", "store": "#f1f1f1"}


def box(ax, x, y, w, h, title, lines, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc=FILL[kind], ec=EDGE, lw=0.8))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=7.6, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.095, "\n".join(lines), ha="center", va="top", fontsize=6.3, color=INK,
            linespacing=1.35)


def arrow(ax, p, q, label=None, dx=0.0, dy=0.0, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=EDGE,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, label, fontsize=5.8, color="#3e4c59",
                ha="center", va="center", bbox=dict(fc="white", ec="none", pad=0.6))


fig = plt.figure(figsize=(3.35, 2.45), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0.03, 0.76)
ax.axis("off")

box(ax, 0.02, 0.50, 0.30, 0.24, "색상 분류 컨베이어", ["TCS34725 RGB 센서", "NEMA17 · Arduino Mega", "빨강·초록 → 불량함"], "plant")
box(ax, 0.35, 0.50, 0.30, 0.24, "저울 + 파란 통", ["파란 큐브 적치", "7-segment 표시부", "USB 카메라 640×480"], "plant")
box(ax, 0.68, 0.50, 0.30, 0.24, "팔레트", ["① 1층  ② 옆", "③ 2층(첫 통 위)", "SO-101 이 적재"], "store")
box(ax, 0.02, 0.05, 0.43, 0.35, "Jetson Orin Nano", ["YOLOv8 · ONNX Runtime", "캡처·추론 스레드 분리",
                                                     "2단계 디지털 줌 · 숫자 조합", "FastAPI 모니터링 (포트 5000)",
                                                     "TCP 트리거 (포트 8765)"], "edge")
box(ax, 0.55, 0.05, 0.43, 0.35, "SO-101 + ACT", ["단계별 정책 π1 · π2 · π3", "입력: 손목·상단 영상 + 관절",
                                                 "출력: 관절 목표 100스텝", "WAIT→RUN→RETURN_HOME",
                                                 "PC 학습 → Orin Nano 실행"], "robot")

arrow(ax, (0.32, 0.62), (0.35, 0.62))
arrow(ax, (0.50, 0.50), (0.30, 0.40), "무게 영상", dx=0.03, dy=0.02)
arrow(ax, (0.455, 0.20), (0.545, 0.20))
ax.text(0.50, 0.26, "≥118g\n'run'", fontsize=5.6, ha="center", va="center", color="#3e4c59")
arrow(ax, (0.80, 0.40), (0.80, 0.50), "적재", dx=0.045)
arrow(ax, (0.64, 0.40), (0.56, 0.50), "통 집기", dx=-0.06, dy=0.0)

fig.savefig("paper/fig1_system.png", dpi=300)
print("saved paper/fig1_system.png")
