"""Figure: 7-segment weight-recognition pipeline (single-column width)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = "Noto Sans CJK JP"
INK, EDGE = "#1f2933", "#52606d"
FILL = {"thread": "#e8f0fb", "yolo": "#fdf1dc", "post": "#e6f4ea", "out": "#f1f1f1"}


def box(ax, x, y, w, h, title, sub, kind, fs=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012", fc=FILL[kind], ec=EDGE, lw=0.7))
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=fs, weight="bold", color=INK)
    if sub:
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=5.6, color=INK, linespacing=1.25)


def arrow(ax, p, q, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=0.7, color=EDGE, ls=ls))


fig = plt.figure(figsize=(3.35, 2.15), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.64)
ax.axis("off")

# thread lanes
ax.add_patch(Rectangle((0.01, 0.47), 0.98, 0.16, fc="none", ec="#c9c8c2", lw=0.6, ls="--"))
ax.text(0.02, 0.615, "캡처 스레드 (최대 속도)", fontsize=5.6, color="#52514e", va="top")
ax.add_patch(Rectangle((0.01, 0.02), 0.98, 0.42, fc="none", ec="#c9c8c2", lw=0.6, ls="--"))
ax.text(0.98, 0.425, "추론 스레드 (5초 주기, ONNX Runtime 6코어)", fontsize=5.6, color="#52514e", va="top", ha="right")

box(ax, 0.05, 0.485, 0.24, 0.10, "USB 카메라", "640×480", "thread")
box(ax, 0.38, 0.485, 0.24, 0.10, "공유 버퍼", "최신 프레임 · Lock", "thread")
box(ax, 0.71, 0.485, 0.24, 0.10, "MJPEG 스트림", "FastAPI /video_feed", "out")
arrow(ax, (0.29, 0.535), (0.38, 0.535))
arrow(ax, (0.62, 0.535), (0.71, 0.535))

box(ax, 0.04, 0.25, 0.20, 0.13, "1단계 YOLO", "전체 프레임\n→ screen 영역", "yolo")
box(ax, 0.28, 0.25, 0.20, 0.13, "디지털 줌", "크롭(+15px)\n→ 640×640", "yolo")
box(ax, 0.52, 0.25, 0.20, 0.13, "2단계 YOLO", "숫자 0~9 탐지", "yolo")
box(ax, 0.76, 0.25, 0.20, 0.13, "숫자 조합", "①클래스 ②NMS\n③구역 ④밀도", "post")
for a, b in ((0.24, 0.28), (0.48, 0.52), (0.72, 0.76)):
    arrow(ax, (a, 0.315), (b, 0.315))
arrow(ax, (0.50, 0.485), (0.14, 0.38))

box(ax, 0.40, 0.05, 0.25, 0.12, "무게 ≥ 118g ?", "", "post", fs=6.4)
box(ax, 0.72, 0.05, 0.24, 0.12, "TCP 트리거", "포트 8765 'run'", "out")
box(ax, 0.04, 0.05, 0.25, 0.12, "대시보드", "/data · 탐지 로그", "out")
arrow(ax, (0.86, 0.25), (0.62, 0.17))
arrow(ax, (0.65, 0.11), (0.72, 0.11))
arrow(ax, (0.40, 0.11), (0.29, 0.11))
ax.text(0.685, 0.135, "예", fontsize=5.6, color="#52514e", ha="center")

fig.savefig("paper/fig_vision.png", dpi=300)
print("saved paper/fig_vision.png")
