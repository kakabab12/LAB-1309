"""지연 설명 그림 (2026-10-02): 계산하는 0.56초 동안 무슨 일이 생기고, 두 기법이 어디를 고치는지."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
fp = "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"
font_manager.fontManager.addfont(fp)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()

fig, axes = plt.subplots(3, 1, figsize=(11, 6.6), sharex=True)
C_OLD, C_NEW, C_FIX, C_COR = "#9aa5b1", "#2a78d6", "#eb6834", "#1baf7a"

def base(ax, title):
    ax.set_ylim(0, 3); ax.set_yticks([]); ax.set_xlim(-1, 42)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, loc="left", fontsize=11)

ax = axes[0]
base(ax, "① 지연이 없다면: 사진을 보는 순간 바로 새 계획")
ax.barh(2, 10, left=0, color=C_NEW, height=0.5); ax.text(5, 2, "계획 1 실행", ha="center", va="center", color="white", fontsize=9)
ax.barh(2, 10, left=10, color=C_NEW, height=0.5, alpha=0.75); ax.text(15, 2, "계획 2 실행", ha="center", va="center", color="white", fontsize=9)
ax.barh(2, 10, left=20, color=C_NEW, height=0.5, alpha=0.55); ax.text(25, 2, "계획 3 실행", ha="center", va="center", color="white", fontsize=9)
for x in (0, 10, 20, 30):
    ax.annotate("사진", (x, 2.35), (x, 2.9), ha="center", fontsize=8, arrowprops=dict(arrowstyle="->", color="#555"))
ax.text(31, 1.0, "시뮬레이터에서만 가능 (1080 Ti 는 계산에 0.56초)", fontsize=9, color="#555")

ax = axes[1]
base(ax, "② 실제 로봇 (지연 11스텝): 계산하는 동안 이전 계획을 계속 따라가고, 도착하면 갈아탄다")
ax.barh(2, 11, left=0, color=C_OLD, height=0.5); ax.text(5.5, 2, "이전 계획 계속", ha="center", va="center", color="white", fontsize=9)
ax.barh(1, 11, left=0, color="#e8e6df", height=0.5, edgecolor="#bbb"); ax.text(5.5, 1, "모델 계산 0.56초", ha="center", va="center", fontsize=9)
ax.barh(2, 12, left=11, color=C_NEW, height=0.5); ax.text(17, 2, "새 계획 (0.56초 전 사진으로 만듦)", ha="center", va="center", color="white", fontsize=9)
ax.annotate("사진", (0, 2.35), (0, 2.9), ha="center", fontsize=8, arrowprops=dict(arrowstyle="->", color="#555"))
ax.annotate("이음매에서 손이 튀거나 망설임", (11, 1.75), (24, 0.6), fontsize=9, color="#c8412c",
            arrowprops=dict(arrowstyle="->", color="#c8412c"))

ax = axes[2]
base(ax, "③ 오늘 넣은 두 기법")
ax.barh(2, 11, left=0, color=C_FIX, height=0.5); ax.text(5.5, 2, "기다리는 동안 할 동작", ha="center", va="center", color="white", fontsize=9)
ax.barh(2, 12, left=11, color=C_NEW, height=0.5); ax.text(17, 2, "그 뒤를 이어서 예측", ha="center", va="center", color="white", fontsize=9)
ax.text(24, 2, "학습 때 지연 흉내내기: 앞부분을 정답으로 고정하고\n뒤만 맞히게 학습 → 이음매가 매끄럽다 (추론 시간 그대로)", fontsize=9, va="center", color=C_FIX)
for x in range(11, 23):
    ax.plot([x + 0.5, x + 0.5], [0.75, 1.25], color=C_COR, lw=2)
ax.text(24, 1.0, "A2C2 보정 네트워크: 매 스텝 최신 사진을 보고\n이번 동작을 조금 고친다 (스텝당 수 ms) → 마지막 1~2cm", fontsize=9, va="center", color=C_COR)
axes[2].set_xlabel("시간 (스텝, 20스텝 = 1초)")
fig.tight_layout()
fig.savefig("outputs/media/latency_explain.png", dpi=130)
print("ok")
