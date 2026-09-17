import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["NanumBarunGothic", "NanumGothic", "DejaVu Sans"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"
BASE, SCRIPT, LORA = "#2a78d6", "#8a8984", "#eb6834"
t = json.load(open("outputs/report/naturalness.json"))
order = [("flush", "flush\n그 자리"), ("keep", "keep"), ("blend", "blend"), ("ret_pos", "ret_pos\n위치 복구"),
         ("retreat", "retreat\n원위치"), ("LoRA 1차 flush", "LoRA 1차\nflush"), ("LoRA 2차 flush", "LoRA 2차\nflush")]
order = [(k, l) for k, l in order if k in t]
col = [LORA if "LoRA" in k else (SCRIPT if k.startswith("ret") else BASE) for k, _ in order]

fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=150, facecolor=SURF)
for ax, (m, title, fmt, scale) in zip(axs, [("detour", "우회 비율 (1 = 곧게 감)", "{:.2f}", 1),
                                           ("path_cm", "전환 → B 성공까지 이동 거리 (cm)", "{:.0f}", 1)]):
    ax.set_facecolor(SURF)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    vals = [t[k][m] * scale for k, _ in order]
    x = np.arange(len(order))
    ax.bar(x, vals, 0.62, color=col, zorder=2)
    for xi, v in zip(x, vals):
        ax.text(xi, v * 1.01, fmt.format(v), ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, [l for _, l in order], fontsize=8, color=INK)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
fig.suptitle("자연스러움: 원위치로 돌아가는 방식은 38cm 더 돌아간다", x=0.01, ha="left", color=INK, fontsize=12)
fig.text(0.01, 0.005, "B 성공 에피소드만, 중앙값. 파랑 = 모델만 사용, 회색 = 스크립트 복귀 포함(진단용), 주황 = LoRA 학습 모델",
         color=INK2, fontsize=8)
fig.tight_layout(rect=[0, 0.03, 1, 0.93])
fig.savefig("outputs/report/naturalness.png", facecolor=SURF)
print("ok")
