#!/usr/bin/env python
"""
실패는 언제 확정되는가? — 전환 이후 시간에 따라 성공/실패 궤적이 갈리는 시점 찾기

왜 묻나: 오라클 실험에서 **같은 상태에서 다시 뽑으면 성공**한다는 것을 확인했다.
        그러면 실패는 상태 탓이 아니라 **뽑힌 궤적** 탓이다.
        "잘못 가고 있다"를 **일찍** 알 수 있으면, 감지해서 다시 뽑는 방법이 성립한다.

무엇을 재나: 전환 이후 매 스텝, 팔 끝이 **B 를 정상 수행할 때 지나가는 길**에서 얼마나 떨어져 있는지.
            (B 의 정상 궤적 = data/v2 의 교란 없는 성공 궤적. 평가 대상과 겹치지 않음)
            성공한 에피소드와 실패한 에피소드의 곡선이 **언제부터** 갈리는지 본다.

판정: 각 스텝에서 Mann-Whitney U 검정 → p < 0.05 가 처음 되는 스텝
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["NanumBarunGothic", "NanumGothic", "DejaVu Sans"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"
WINDOW = 120  # 전환 이후 몇 스텝까지 볼지 (20스텝 = 1초)


def normal_trees():
    """태스크별 '교란 없는 정상 궤적' KD 트리."""
    pts = defaultdict(list)
    for f in Path("data/v2/episodes").glob("T*.npz"):
        z = np.load(f, allow_pickle=True)
        if float(z["offset_cm"]) == 0.0:
            pts[str(z["task"])].append(z["eef_pos"])
    return {k: cKDTree(np.concatenate(v)) for k, v in pts.items()}


def main():
    trees = normal_trees()
    print("정상 궤적이 있는 태스크:", len(trees))
    curves = {True: [], False: []}
    groups = {}
    for f in glob.glob("outputs/switch/A*_B*_grasp*_flush.json"):
        d = json.load(open(f))
        a = d["args"]
        if a.get("latency_steps", 0):
            continue
        for e in d["episodes"]:
            if not e.get("switched"):
                continue
            lang = e.get("task_b")
            if lang not in trees:
                continue
            tag = (f"A{a['task_a']}_B{a['task_b']}_{a['switch_at'].replace(':', '')}"
                   f"_flush_ep{e['episode']}")
            p = Path("outputs/switch/traj") / f"{tag}.npz"
            if not p.exists():
                continue
            z = np.load(p)
            ph = z["phase"].astype(str)
            idx = np.where(ph == "B")[0]
            if len(idx) == 0:
                continue
            seg = z["pos"][idx[0]:idx[0] + WINDOW]
            dist, _ = trees[lang].query(seg)
            c = np.full(WINDOW, np.nan)
            c[:len(dist)] = dist * 100  # cm
            curves[bool(e.get("b_success"))].append(c)
            groups.setdefault((a["task_a"], a["task_b"], a["switch_at"]), {True: [], False: []})[
                bool(e.get("b_success"))].append(c)

    S = np.array(curves[True])
    F = np.array(curves[False])
    print(f"성공 {len(S)}개, 실패 {len(F)}개 에피소드")
    if len(S) < 5 or len(F) < 5:
        print("표본이 부족합니다."); return

    first_sig = None
    ps = np.ones(WINDOW)
    for t in range(WINDOW):
        s, f = S[:, t][~np.isnan(S[:, t])], F[:, t][~np.isnan(F[:, t])]
        if len(s) >= 5 and len(f) >= 5:
            ps[t] = mannwhitneyu(s, f, alternative="two-sided").pvalue
            if first_sig is None and ps[t] < 0.05:
                first_sig = t

    ms, mf = np.nanmedian(S, axis=0), np.nanmedian(F, axis=0)
    print(f"\n{'스텝':>6s}{'초':>6s}{'성공(cm)':>10s}{'실패(cm)':>10s}{'p':>10s}")
    for t in [0, 10, 20, 30, 40, 60, 80, 100, WINDOW - 1]:
        print(f"{t:6d}{t / 20:6.1f}{ms[t]:10.1f}{mf[t]:10.1f}{ps[t]:10.3f}")
    if first_sig is not None:
        print(f"\n처음 유의하게 갈리는 시점: **{first_sig} 스텝 ({first_sig / 20:.1f}초)**")
        print(f"  그때 성공 {ms[first_sig]:.1f}cm vs 실패 {mf[first_sig]:.1f}cm")
        direction = "실패가 더 멀다" if mf[first_sig] > ms[first_sig] else "**성공이 더 멀다 (반대 방향!)**"
        print(f"  방향: {direction}")
        print("  ⚠️ 이 전체 비교는 쉬운 쌍(8→7, 4→7)과 어려운 쌍(1→5, 2→5)이 섞여 있어 교란된다.")
        print("     아래 '쌍 안에서' 비교를 봐야 한다.")
    else:
        print("\n끝까지 유의하게 갈리지 않았다 → 이 지표로는 실패를 미리 알 수 없다")

    # ---- 쌍 안에서 다시 (쉬운 쌍/어려운 쌍이 섞이면 결론이 뒤집힌다: Simpson's paradox) ----
    print("\n== 같은 태스크 쌍·같은 전환 시점 안에서만 비교")
    per_group = []
    for k, v in sorted(groups.items()):
        s, f = np.array(v[True]), np.array(v[False])
        if len(s) < 3 or len(f) < 3:
            continue
        ds = np.nanmedian(f, axis=0) - np.nanmedian(s, axis=0)  # 실패 - 성공 (양수면 실패가 더 멀다)
        per_group.append((k, len(s), len(f), ds))
        print(f"  A{k[0]}→B{k[1]} {k[2]}: 성공 {len(s)}, 실패 {len(f)} | "
              f"실패−성공 거리차(cm) 0.5초 {ds[10]:+.1f}, 1.5초 {ds[30]:+.1f}, 3초 {ds[60]:+.1f}")
    if per_group:
        D = np.array([g[3] for g in per_group])
        med = np.nanmedian(D, axis=0)
        pos = [(int(np.nansum(D[:, t] > 0)), len(D)) for t in [10, 30, 60]]
        print(f"  집단 {len(per_group)}개 평균: 0.5초 {med[10]:+.1f}cm, 1.5초 {med[30]:+.1f}cm, 3초 {med[60]:+.1f}cm")
        print(f"  '실패가 더 멀다'인 집단 수: 0.5초 {pos[0][0]}/{pos[0][1]}, "
              f"1.5초 {pos[1][0]}/{pos[1][1]}, 3초 {pos[2][0]}/{pos[2][1]}")
        print("  → 집단 안에서도 일정한 방향이 없으면, 이 지표로는 실패를 조기에 감지할 수 없다")
    else:
        print("  성공·실패가 각각 3개 이상인 집단이 없습니다 (쉬운 쌍은 거의 성공, 어려운 쌍은 거의 실패)")

    out = {"n_success": len(S), "n_fail": len(F), "first_sig_step": first_sig,
           "per_group": [{"pair": f"A{g[0][0]}→B{g[0][1]}", "switch_at": g[0][2],
                          "n_s": g[1], "n_f": g[2],
                          "diff_0.5s": round(float(g[3][10]), 2),
                          "diff_1.5s": round(float(g[3][30]), 2),
                          "diff_3s": round(float(g[3][60]), 2)} for g in per_group],
           "median_success": [None if np.isnan(x) else round(float(x), 2) for x in ms],
           "median_fail": [None if np.isnan(x) else round(float(x), 2) for x in mf],
           "p": [round(float(x), 4) for x in ps]}
    Path("outputs/report").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("outputs/report/divergence_time.json", "w"), ensure_ascii=False, indent=2)

    # ---- 그래프 ----
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150, facecolor=SURF)
    ax.set_facecolor(SURF)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(INK2); ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    x = np.arange(WINDOW) / 20
    for arr, lab, col in [(S, f"B 성공 (n={len(S)})", "#1baf7a"), (F, f"B 실패 (n={len(F)})", "#eb6834")]:
        med = np.nanmedian(arr, axis=0)
        q1, q3 = np.nanpercentile(arr, 25, axis=0), np.nanpercentile(arr, 75, axis=0)
        ax.fill_between(x, q1, q3, color=col, alpha=0.15, lw=0)
        ax.plot(x, med, color=col, lw=2, label=lab)
    if first_sig is not None:
        ax.axvline(first_sig / 20, color=INK2, ls="--", lw=1)
        ax.text(first_sig / 20 + 0.05, ax.get_ylim()[1] * 0.95,
                f"{first_sig / 20:.1f}초부터 갈림", color=INK2, fontsize=9, va="top")
    ax.set_xlabel("전환 이후 시간 (초)", color=INK2)
    ax.set_ylabel("B 의 정상 경로와의 거리 (cm)", color=INK2)
    ax.set_title("실패는 언제 확정되는가", color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    fig.text(0.01, 0.01, "그 자리 전환(flush), 중앙값과 사분위 범위. B 정상 경로는 교란 없는 성공 궤적(평가와 분리)",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig("outputs/report/divergence_time.png", facecolor=SURF)
    print("저장: outputs/report/divergence_time.png")


if __name__ == "__main__":
    main()
