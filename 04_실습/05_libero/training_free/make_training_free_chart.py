#!/usr/bin/env python
"""학습 없는 전환 전략 비교 그래프: B 성공률(+Wilson CI) 과 우회 비율."""
import json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["NanumBarunGothic", "NanumGothic", "DejaVu Sans"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"
STRATS = [("flush", "flush\n그 자리 전환", "#2a78d6"), ("rtc", "rtc\n이전 동작 이어붙임", "#eb6834"),
          ("flush_rtc", "flush_rtc\n끊고 나서 이어붙임", "#1baf7a"), ("bon", "bon\n후보 8개 중 선택", "#eda100"),
          ("retreat", "retreat\n원위치(진단용)", "#8a8984")]


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


rates, los, his, labels, cols, ns = [], [], [], [], [], []
for s, lab, col in STRATS:
    eps = []
    for f in glob.glob("outputs/switch/A*_B*_grasp*.json"):
        d = json.load(open(f))
        if d["args"]["strategy"] == s:  # 파일 이름 패턴은 rtc 가 flush_rtc 까지 잡으므로 인자로 거름
            eps += [e for e in d["episodes"] if e.get("switched")]
    if not eps:
        continue
    k = sum(bool(e.get("b_success")) for e in eps); n = len(eps)
    lo, hi = wilson(k, n)
    rates.append(100 * k / n); los.append(100 * lo); his.append(100 * hi); labels.append(lab); cols.append(col); ns.append(n)

nat = json.load(open("outputs/report/naturalness.json"))
fig, axs = plt.subplots(1, 2, figsize=(11, 4.3), dpi=150, facecolor=SURF, gridspec_kw={"width_ratios": [1.2, 1]})
for ax in axs:
    ax.set_facecolor(SURF)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(INK2); ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
x = np.arange(len(labels))
axs[0].bar(x, rates, 0.62, color=cols, zorder=2)
axs[0].errorbar(x, rates, yerr=[np.array(rates) - los, np.array(his) - rates], fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
for xi, r, h, n in zip(x, rates, his, ns):
    axs[0].text(xi, h + 2, f"{r:.0f}%", ha="center", fontsize=9, color=INK2)
axs[0].set_xticks(x, labels, fontsize=8, color=INK); axs[0].set_ylim(0, 105)
axs[0].set_ylabel("B 성공률 (%)", color=INK2)
axs[0].set_title("전환 후 B 성공률", color=INK, loc="left", fontsize=11)
keys = [s for s, _, _ in STRATS if s in nat]
det = [nat[s]["detour"] for s in keys]
colmap = {s: c for s, _, c in STRATS}
axs[1].bar(np.arange(len(keys)), det, 0.62, color=[colmap[s] for s in keys], zorder=2)
for xi, v in enumerate(det):
    axs[1].text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=9, color=INK2)
axs[1].set_xticks(np.arange(len(keys)), keys, fontsize=9, color=INK)
axs[1].set_title("우회 비율 (1 = 곧게 감, B 성공 에피소드)", color=INK, loc="left", fontsize=11)
fig.suptitle("학습 없는 전환 전략: 어느 것도 flush 를 확실히 넘지 못했다", x=0.01, ha="left", color=INK, fontsize=12)
fig.text(0.01, 0.005, "4개 태스크 쌍 × 잡은 직후·들고 이동 중 × 10회. 오차막대 = Wilson 95% 신뢰구간. retreat 은 초기 자세 복귀라 진단용",
         color=INK2, fontsize=8)
fig.tight_layout(rect=[0, 0.03, 1, 0.93])
fig.savefig("outputs/report/training_free_strategies.png", facecolor=SURF)
print("ok", dict(zip([s for s, _, _ in STRATS], rates)))
