#!/usr/bin/env python
"""ablation + LoRA 1차 결과 그래프 (일지용)."""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["NanumBarunGothic", "NanumGothic", "DejaVu Sans"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
OUT = Path("outputs/report"); OUT.mkdir(parents=True, exist_ok=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def style(ax):
    ax.set_facecolor(SURF)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def episodes(d, strategy, timings=("grasp:3", "grasp:20")):
    eps = []
    for f in glob.glob(f"{d}/A*_{strategy}.json"):
        for e in json.load(open(f))["episodes"]:
            if e.get("switched") and e["switch_at"] in timings:
                eps.append(e)
    return eps


# 1) ablation: B 성공 / A 재개
order = [("flush", "그 자리에서\n지시만 교체"), ("release", "내려놓기만"), ("ret_rot", "내려놓고\n회전만 복구"),
         ("ret_pos", "내려놓고\n위치만 복구"), ("retreat", "위치+회전\n모두 복구")]
fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=150, facecolor=SURF)
style(ax)
x = np.arange(len(order)); w = 0.36
for j, (key, label, col) in enumerate([("b_success", "B 성공", C1), ("a_resume_success", "A 재개", C3)]):
    ps, lo, hi = [], [], []
    for s, _ in order:
        v = [bool(e.get(key)) for e in episodes("outputs/switch", s)]
        k, n = sum(v), len(v)
        ps.append(100 * k / n); l, h = wilson(k, n); lo.append(100 * l); hi.append(100 * h)
    xs = x + (j - 0.5) * (w + 0.02)
    ax.bar(xs, ps, w, color=col, label=label, zorder=2)
    ax.errorbar(xs, ps, yerr=[np.array(ps) - lo, np.array(hi) - ps], fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
    for xi, p, h in zip(xs, ps, hi):
        ax.text(xi, h + 1.5, f"{p:.0f}", ha="center", va="bottom", fontsize=8, color=INK2)
ax.set_xticks(x, [l for _, l in order], color=INK, fontsize=9)
ax.set_ylim(0, 108); ax.set_ylabel("성공률 (%)", color=INK2)
ax.set_title("retreat 을 쪼개보니: 핵심은 '위치' 복구", color=INK, loc="left", fontsize=12)
ax.legend(frameon=False, loc="upper left", labelcolor=INK)
fig.text(0.01, 0.005, "잡은 직후 + 들고 이동 중 전환, 4개 태스크 쌍, 조건당 약 72회. 오차막대 = Wilson 95% 신뢰구간", color=INK2, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ablation.png", facecolor=SURF); plt.close(fig)

# 2) LoRA 1차: 전환(flush) B 성공 전후
pairs = [("A8_B7", "8→7"), ("A4_B7", "4→7"), ("A1_B5", "1→5"), ("A2_B5", "2→5")]
tims = [("grasp3", "잡은 직후"), ("grasp20", "들고 이동 중")]
labels, before, after = [], [], []
for pk, pl in pairs:
    for tk, tl in tims:
        b = json.load(open(f"outputs/switch/{pk}_{tk}_flush.json"))["summary"]
        a = json.load(open(f"outputs/switch_lora/{pk}_{tk}_flush.json"))["summary"]
        labels.append(f"{pl}\n{tl}"); before.append(100 * b.get("b_success_rate", 0)); after.append(100 * a.get("b_success_rate", 0))
fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150, facecolor=SURF)
style(ax)
x = np.arange(len(labels)); w = 0.38
ax.bar(x - w / 2 - 0.01, before, w, color="#8a8984", label="원본 SmolVLA", zorder=2)
ax.bar(x + w / 2 + 0.01, after, w, color=C1, label="LoRA 1차", zorder=2)
for xi, b, a in zip(x, before, after):
    ax.text(xi - w / 2, b + 1.5, f"{b:.0f}", ha="center", fontsize=8, color=INK2)
    ax.text(xi + w / 2, a + 1.5, f"{a:.0f}", ha="center", fontsize=8, color=INK2)
ax.set_xticks(x, labels, fontsize=8.5, color=INK); ax.set_ylim(0, 100)
ax.set_ylabel("B 성공률 (%)", color=INK2)
ax.set_title("LoRA 1차: 잡은 직후만 오르고, 들고 이동 중은 내리고, 1→5·2→5는 0%", color=INK, loc="left", fontsize=12)
ax.legend(frameon=False, loc="upper right", labelcolor=INK)
fig.text(0.01, 0.005, "그 자리에서 전환(flush), 조건당 10회. 합계 32% → 32%", color=INK2, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "lora_v1_switch.png", facecolor=SURF); plt.close(fig)

# 3) LoRA 1차 학습 곡선
h = json.load(open("outputs/lora_v1/history.json"))
tr = [(r["step"], r["loss"]) for r in h if "loss" in r]
va = [(r["step"], r["val_loss"]) for r in h if "val_loss" in r]
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150, facecolor=SURF)
style(ax)
ax.plot(*zip(*tr), color=C1, lw=1.5, label="학습 loss (25스텝 평균)")
ax.plot(*zip(*va), color=C2, lw=2, marker="o", ms=7, label="검증 loss")
ax.set_xlabel("스텝", color=INK2); ax.set_ylabel("flow matching loss", color=INK2)
ax.set_title("LoRA 1차 학습 곡선 — 학습 자체는 정상", color=INK, loc="left", fontsize=12)
ax.legend(frameon=False, labelcolor=INK)
fig.text(0.01, 0.005, "GTX 1080 Ti fp32, 배치 4 × 누적 2, VRAM 2.67GB, 2000스텝 32분", color=INK2, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "lora_v1_curve.png", facecolor=SURF); plt.close(fig)

# 4) 자세 민감도 전후 (태스크 1: 가장 크게 나빠짐)
fig, axs = plt.subplots(1, 2, figsize=(9, 3.8), dpi=150, facecolor=SURF)
for ax, (axis, vals, xl) in zip(axs, [("offset", [0, 4, 8], "위치 오프셋 (cm)"), ("yaw", [0, 10, 20, 30], "손목 회전 (도)")]):
    style(ax)
    for src, col, lab in [("outputs/pose", "#8a8984", "원본"), ("outputs/pose_lora", C1, "LoRA 1차")]:
        ys = []
        for v in vals:
            name = f"T1_off{v}_yaw0" if axis == "offset" else f"T1_off0_yaw{v}"
            ys.append(100 * json.load(open(f"{src}/{name}.json"))["summary"]["success_rate"])
        ax.plot(vals, ys, color=col, lw=2, marker="o", ms=7, label=lab)
    ax.set_xticks(vals); ax.set_ylim(-3, 105); ax.set_xlabel(xl, color=INK2)
axs[0].set_ylabel("성공률 (%)", color=INK2)
axs[0].legend(frameon=False, labelcolor=INK)
fig.suptitle("태스크 1 (put the bowl on the stove): 학습 후 교란 없는 상태에서도 100% → 60%", x=0.01, ha="left", color=INK, fontsize=12)
fig.text(0.01, 0.005, "조건당 10회. 학습 데이터의 40%가 같은 스토브 근처의 turn on the stove 라 간섭한 것으로 추정 (이 태스크는 10.6%)", color=INK2, fontsize=8)
fig.tight_layout(rect=[0, 0.03, 1, 0.94]); fig.savefig(OUT / "lora_v1_pose_task1.png", facecolor=SURF); plt.close(fig)
print("ok")
