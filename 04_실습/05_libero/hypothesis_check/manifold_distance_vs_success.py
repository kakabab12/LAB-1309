#!/usr/bin/env python
"""
가설 검증: 전환 성공을 가르는 건 '초기 자세와의 거리'가 아니라 'B 를 정상 수행할 때 팔이 지나가는 길과의 거리'인가?

B 의 정상 경로 = data/v2 의 교란 없는 정상 궤적(offset 0) 중 B 태스크 것만 사용.
(flush 성공 궤적은 쓰지 않음 — 평가 대상과 겹치면 답을 미리 보는 셈)
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["NanumBarunGothic", "NanumGothic", "DejaVu Sans"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"

normal = defaultdict(list)
for f in Path("data/v2/episodes").glob("T*.npz"):
    z = np.load(f, allow_pickle=True)
    if float(z["offset_cm"]) == 0.0:
        normal[str(z["task"])].append(z["eef_pos"])
trees = {t: cKDTree(np.concatenate(v)) for t, v in normal.items()}
print("정상 경로가 있는 태스크:", {t: len(v) for t, v in normal.items()})

rows = []
for f in Path("outputs/switch").glob("A*_B*_*_flush.json"):
    d = json.load(open(f)); a = d["args"]
    for e in d["episodes"]:
        if not e.get("switched") or e["task_b"] not in trees:
            continue
        tag = f"A{a['task_a']}_B{a['task_b']}_{a['switch_at'].replace(':', '')}_flush_ep{e['episode']}"
        z = np.load(f"outputs/switch/traj/{tag}.npz")
        ts = e["a_steps_before_switch"]
        p = z["pos"][ts - 1]
        d_home = float(np.linalg.norm(p - z["pos"][0]) * 100)
        d_b, _ = trees[e["task_b"]].query(p)
        rows.append({"d_home": d_home, "d_b": float(d_b * 100), "ok": bool(e.get("b_success")),
                     "pair": f"{a['task_a']}→{a['task_b']}", "at": a["switch_at"]})

dh = np.array([r["d_home"] for r in rows]); db = np.array([r["d_b"] for r in rows]); ok = np.array([r["ok"] for r in rows], float)
r_home = float(np.corrcoef(dh, ok)[0, 1]); r_b = float(np.corrcoef(db, ok)[0, 1])
print(f"에피소드 {len(rows)}개")
print(f"  초기 자세와의 거리   vs 성공: r = {r_home:+.2f}")
print(f"  B 정상 경로와의 거리 vs 성공: r = {r_b:+.2f}")

def binned(x, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum():
            out.append((f"{lo:g}~{hi:g}", 100 * ok[m].mean(), int(m.sum())))
    return out

qs = np.quantile(db, [0, 0.25, 0.5, 0.75, 1.0]); qs[-1] += 1e-6
b_bins = binned(db, np.round(qs, 1))
print("  B 경로 거리 구간별:", [(l, f"{p:.0f}%", n) for l, p, n in b_bins])

# 태스크 쌍 안에서도 같은 경향인지 (쌍별 차이가 원인일 수 있으므로)
within = []
for pair in sorted({r["pair"] for r in rows}):
    sub = [r for r in rows if r["pair"] == pair]
    o = np.array([r["ok"] for r in sub], float)
    if 0 < o.mean() < 1 and len(sub) > 5:
        within.append((pair, float(np.corrcoef([r["d_b"] for r in sub], o)[0, 1]),
                       float(np.corrcoef([r["d_home"] for r in sub], o)[0, 1]), len(sub)))
for pair, rb, rh, n in within:
    print(f"  쌍 {pair} (n={n}): B경로거리 r={rb:+.2f}, 초기자세거리 r={rh:+.2f}")

fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=150, facecolor=SURF)
for ax, x, name, rr in [(axs[0], dh, "초기 자세와의 거리 (cm)", r_home), (axs[1], db, "B 정상 경로와의 거리 (cm)", r_b)]:
    ax.set_facecolor(SURF)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    jitter = (np.random.default_rng(0).random(len(ok)) - 0.5) * 0.12
    ax.scatter(x[ok == 1], (ok + jitter)[ok == 1], s=16, color="#1baf7a", alpha=0.7, label="B 성공")
    ax.scatter(x[ok == 0], (ok + jitter)[ok == 0], s=16, color="#eb6834", alpha=0.7, label="B 실패")
    ax.set_yticks([0, 1], ["실패", "성공"], color=INK)
    ax.set_xlabel(name, color=INK2)
    ax.set_title(f"상관 r = {rr:+.2f}", color=INK, loc="left", fontsize=11)
axs[0].legend(frameon=False, loc="center right", labelcolor=INK)
fig.suptitle("전환 성공을 가르는 거리는 어느 쪽인가", x=0.01, ha="left", color=INK, fontsize=12)
fig.text(0.01, 0.005, f"그 자리 전환(flush) {len(rows)}회. B 정상 경로 = 교란 없이 B 를 성공한 궤적의 팔 끝 위치", color=INK2, fontsize=8)
fig.tight_layout(rect=[0, 0.03, 1, 0.94])
fig.savefig("outputs/report/manifold_distance_vs_success.png", facecolor=SURF)
json.dump({"n": len(rows), "r_home": r_home, "r_b_manifold": r_b, "bins_b": b_bins, "within_pair": within},
          open("outputs/report/manifold_distance_vs_success.json", "w"), ensure_ascii=False, indent=2)
