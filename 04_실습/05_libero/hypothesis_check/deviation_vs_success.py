#!/usr/bin/env python
"""전환 순간 팔이 초기 자세에서 얼마나 떨어져 있었나 vs 그 자리 전환(flush) B 성공 — 에피소드 단위."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["NanumBarunGothic", "NanumGothic", "DejaVu Sans"]
SURF, INK, INK2, GRID, C1 = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd", "#2a78d6"

rows = []
for f in Path("outputs/switch").glob("A*_B*_*_flush.json"):
    d = json.load(open(f)); a = d["args"]
    for e in d["episodes"]:
        if not e.get("switched"):
            continue
        tag = f"A{a['task_a']}_B{a['task_b']}_{a['switch_at'].replace(':', '')}_flush_ep{e['episode']}"
        z = np.load(f"outputs/switch/traj/{tag}.npz")
        ts = e["a_steps_before_switch"]
        dev = float(np.linalg.norm(z["pos"][ts - 1] - z["pos"][0]) * 100)
        rows.append((dev, bool(e.get("b_success")), f"{a['task_a']}→{a['task_b']}", a["switch_at"]))

dev = np.array([r[0] for r in rows]); ok = np.array([r[1] for r in rows])
bins = [0, 10, 15, 20, 25, 35]
labels, rates, ns, los, his = [], [], [], [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (dev >= lo) & (dev < hi)
    n = int(m.sum()); k = int(ok[m].sum())
    if n == 0:
        continue
    p = k / n; z = 1.96; dd = 1 + z * z / n
    c = (p + z * z / (2 * n)) / dd; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / dd
    labels.append(f"{lo}~{hi}cm"); rates.append(100 * p); ns.append(n); los.append(100 * (c - h)); his.append(100 * (c + h))

# 점-이연 상관 (이탈 거리 vs 성공)
r = float(np.corrcoef(dev, ok.astype(float))[0, 1])
print(f"에피소드 {len(rows)}개, 이탈 거리-성공 상관 r = {r:.2f}")
for l, p, n in zip(labels, rates, ns):
    print(f"  {l:8s} n={n:3d}  B 성공 {p:.0f}%")

fig, ax = plt.subplots(figsize=(7.2, 4), dpi=150, facecolor=SURF)
ax.set_facecolor(SURF)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(INK2); ax.tick_params(colors=INK2, length=0)
ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
x = np.arange(len(labels))
ax.bar(x, rates, 0.6, color=C1, zorder=2)
ax.errorbar(x, rates, yerr=[np.array(rates) - los, np.array(his) - rates], fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
for xi, p, n, h in zip(x, rates, ns, his):
    ax.text(xi, h + 2, f"{p:.0f}%\n(n={n})", ha="center", va="bottom", fontsize=8, color=INK2)
ax.set_xticks(x, labels, color=INK); ax.set_ylim(0, 115)
ax.set_xlabel("전환 순간 팔 끝이 초기 자세에서 떨어진 거리", color=INK2)
ax.set_ylabel("그 자리 전환(flush) B 성공률 (%)", color=INK2)
ax.set_title(f"멀리 있을수록 실패한다 (상관 r = {r:.2f})", color=INK, loc="left", fontsize=12)
fig.text(0.01, 0.005, f"9/14 flush 에피소드 {len(rows)}개, 4개 태스크 쌍 × 3개 전환 시점. 오차막대 = Wilson 95% 신뢰구간", color=INK2, fontsize=8)
fig.tight_layout(); fig.savefig("outputs/report/deviation_vs_success.png", facecolor=SURF)
json.dump({"n": len(rows), "r": r, "bins": dict(zip(labels, zip(rates, ns)))}, open("outputs/report/deviation_vs_success.json", "w"), ensure_ascii=False, indent=2)
