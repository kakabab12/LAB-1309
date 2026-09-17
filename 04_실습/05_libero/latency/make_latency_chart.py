import json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["NanumBarunGothic", "NanumGothic", "DejaVu Sans"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"

def load(pat):
    out = {}
    for f in glob.glob(pat):
        d = json.load(open(f)); a = d["args"]
        out[(a["task_a"], a.get("task_b"), a["strategy"], a.get("latency_steps", 0))] = d["episodes"]
    return out
L = load("outputs/latency/*.json"); L.update(load("outputs/rtc_indist/*.json"))

def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

lats = [0, 5, 10, 20]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150, facecolor=SURF, gridspec_kw={"width_ratios": [1.3, 1]})
for ax in axs:
    ax.set_facecolor(SURF)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(INK2); ax.spines["left"].set_color(INK2); ax.tick_params(colors=INK2)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
for s, lab, col in [("none", "RTC 끔", "#2a78d6"), ("rtc_all", "RTC 켬", "#eb6834")]:
    ys, lo, hi = [], [], []
    for lat in lats:
        k = n = 0
        for t in [1, 5, 7, 8]:
            eps = L.get((t, None, s, lat), []); k += sum(bool(e.get("a_success")) for e in eps); n += len(eps)
        ys.append(100 * k / n); l, h = wilson(k, n); lo.append(100 * l); hi.append(100 * h)
    x = [v / 20 for v in lats]
    axs[0].fill_between(x, lo, hi, color=col, alpha=0.12, lw=0)
    axs[0].plot(x, ys, color=col, lw=2, marker="o", ms=7, label=lab)
    for xi, yi in zip(x, ys):
        axs[0].text(xi, yi + 3, f"{yi:.0f}%", ha="center", fontsize=8, color=col)
axs[0].set_xticks([v / 20 for v in lats]); axs[0].set_ylim(-3, 105)
axs[0].set_xlabel("추론 지연 (초)", color=INK2); axs[0].set_ylabel("태스크 성공률 (%)", color=INK2)
axs[0].set_title("전환 없이 태스크 하나만 수행", color=INK, loc="left", fontsize=11)
axs[0].legend(frameon=False, labelcolor=INK)
names = [("flush", "flush\n멈추고 기다림"), ("keep", "keep\n이전 계획 계속"), ("rtc", "rtc")]
x = np.arange(len(names)); w = 0.36
for j, (lat, lab, col) in enumerate([(0, "지연 없음", "#8a8984"), (10, "지연 0.5초", "#2a78d6")]):
    ys = []
    for s, _ in names:
        eps = []
        for p in [(8, 7), (4, 7), (1, 5)]:
            if lat == 0:
                try: eps += [e for e in json.load(open(f"outputs/switch/A{p[0]}_B{p[1]}_grasp3_{s}.json"))["episodes"] if e.get("switched")]
                except FileNotFoundError: pass
            else:
                eps += [e for e in L.get((p[0], p[1], s, lat), []) if e.get("switched")]
        ys.append(100 * sum(bool(e.get("b_success")) for e in eps) / max(len(eps), 1))
    axs[1].bar(x + (j - 0.5) * (w + 0.02), ys, w, color=col, label=lab, zorder=2)
    for xi, yi in zip(x + (j - 0.5) * (w + 0.02), ys):
        axs[1].text(xi, yi + 1, f"{yi:.0f}", ha="center", fontsize=8, color=INK2)
axs[1].set_xticks(x, [n for _, n in names], fontsize=8, color=INK); axs[1].set_ylim(0, 45)
axs[1].set_ylabel("B 성공률 (%)", color=INK2)
axs[1].set_title("잡은 직후 전환 (3쌍 × 10회)", color=INK, loc="left", fontsize=11)
axs[1].legend(frameon=False, labelcolor=INK, fontsize=8)
fig.suptitle("추론 지연: 0.5초만 돼도 무너지고, RTC 는 지연이 있을 때만 도움", x=0.01, ha="left", color=INK, fontsize=12)
fig.text(0.01, 0.005, "태스크 1·5·7·8 × 10회. 띠 = Wilson 95% 신뢰구간. 지연 모사: 요청한 동작이 L 스텝 뒤 도착, 그동안 이전 계획 실행", color=INK2, fontsize=8)
fig.tight_layout(rect=[0, 0.03, 1, 0.93]); fig.savefig("outputs/report/latency.png", facecolor=SURF); print("ok")
