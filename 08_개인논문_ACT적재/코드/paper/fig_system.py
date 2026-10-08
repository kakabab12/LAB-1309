"""Figure 1: overall system diagram (single-column width). FIG_LANG=en writes the English version (blind review file)."""

import os

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

EN = os.environ.get("FIG_LANG") == "en"
T = (dict(conv=("Sorting conveyor", ["TCS34725 RGB sensor", "NEMA17 · Arduino Mega", "red·green → reject bin"]),
          scale=("Scale + blue bin", ["blue cubes collected", "7-segment display", "USB camera 640×480"]),
          pallet=("Pallet", ["① floor 1  ② beside", "③ floor 2 (on bin ①)", "stacked by SO-101"]),
          edge=["YOLOv8 · ONNX Runtime", "capture / inference threads", "2-stage digital zoom · digit merge",
                "FastAPI monitor (port 5000)", "TCP trigger (port 8765)"],
          robot=["stage policies π1 · π2 · π3", "in: wrist + top images, joints", "out: 100-step joint chunk",
                 "WAIT→RUN→RETURN_HOME", "trained on PC → run on Orin Nano"],
          weigh="weight image", stack="stack", pick="pick bin")
     if EN else
     dict(conv=("색상 분류 컨베이어", ["TCS34725 RGB 센서", "NEMA17 · Arduino Mega", "빨강·초록 → 불량함"]),
          scale=("저울 + 파란 통", ["파란 큐브 적치", "7-segment 표시부", "USB 카메라 640×480"]),
          pallet=("팔레트", ["① 1층  ② 옆", "③ 2층(첫 통 위)", "SO-101 이 적재"]),
          edge=["YOLOv8 · ONNX Runtime", "캡처·추론 스레드 분리", "2단계 디지털 줌 · 숫자 조합",
                "FastAPI 모니터링 (포트 5000)", "TCP 트리거 (포트 8765)"],
          robot=["단계별 정책 π1 · π2 · π3", "입력: 손목·상단 영상 + 관절", "출력: 관절 목표 100스텝",
                 "WAIT→RUN→RETURN_HOME", "PC 학습 → Orin Nano 실행"],
          weigh="무게 영상", stack="적재", pick="통 집기"))

box(ax, 0.02, 0.50, 0.30, 0.24, *T["conv"], "plant")
box(ax, 0.35, 0.50, 0.30, 0.24, *T["scale"], "plant")
box(ax, 0.68, 0.50, 0.30, 0.24, *T["pallet"], "store")
box(ax, 0.02, 0.05, 0.43, 0.35, "Jetson Orin Nano", T["edge"], "edge")
box(ax, 0.55, 0.05, 0.43, 0.35, "SO-101 + ACT", T["robot"], "robot")

arrow(ax, (0.32, 0.62), (0.35, 0.62))
arrow(ax, (0.50, 0.50), (0.30, 0.40), T["weigh"], dx=0.03, dy=0.02)
arrow(ax, (0.455, 0.20), (0.545, 0.20))
ax.text(0.50, 0.26, "≥118g\n'run'", fontsize=5.6, ha="center", va="center", color="#3e4c59")
arrow(ax, (0.80, 0.40), (0.80, 0.50), T["stack"], dx=0.045)
arrow(ax, (0.64, 0.40), (0.56, 0.50), T["pick"], dx=-0.06, dy=0.0)

out = "paper/fig1_system_en.png" if EN else "paper/fig1_system.png"
fig.savefig(out, dpi=300)
print("saved", out)
