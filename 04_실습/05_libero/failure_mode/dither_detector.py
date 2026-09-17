#!/usr/bin/env python
"""
"제자리 왕복(dithering)"으로 실패를 조기 감지할 수 있는가?

발견한 것 (2026-09-18)
  실패한 에피소드는 **얼어붙는 게 아니라** 좁은 영역에서 왕복한다.
    뒤 100스텝: 실패는 경로 24.8cm 를 **6.2cm 상자** 안에서 움직임
                성공은 경로 30.0cm 를 **17.6cm 범위**로 뻗어 나감
  거리 기반 지표(B 정상 경로와의 거리)로는 실패를 미리 알 수 없었는데,
  이 지표는 **움직임의 모양**을 보므로 다를 수 있다.

지표: dither = (창 안의 경로 길이) / (창 안의 이동 범위)
      곧게 가면 1 에 가깝고, 제자리 왕복이면 커진다.

재는 것: 전환 이후 시간 t 에서 이 지표로 "결국 실패할지"를 맞히는 AUC.
        일찍(2~3초) AUC 가 높으면 **감지해서 다시 뽑는 방법**이 성립한다.
"""
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["NanumBarunGothic", "NanumGothic", "DejaVu Sans"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"
WIN = 40        # 창 길이 (2초)
MAX_T = 240     # 전환 이후 12초까지
MIN_RANGE_CM = 0.5


def auc(scores, labels):
    """labels: 1 = 실패. 랭크 기반 AUC."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    if y.sum() == 0 or y.sum() == len(y):
        return None, len(y)
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), len(y)


def dither_series(pos):
    """각 시점 t 에서 직전 WIN 스텝의 (경로 길이 / 이동 범위)."""
    out = np.full(len(pos), np.nan)
    for t in range(WIN, len(pos)):
        w = pos[t - WIN:t] * 100
        path = float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum())
        rng = float(np.linalg.norm(w.max(axis=0) - w.min(axis=0)))
        out[t] = path / max(rng, MIN_RANGE_CM)
    return out


def main():
    eps = []
    for f in glob.glob("outputs/switch/A*_B*_grasp*_flush.json"):
        d = json.load(open(f))
        a = d["args"]
        if a.get("latency_steps", 0) or a.get("switch_n_action_steps", 0):
            continue
        for e in d["episodes"]:
            if not e.get("switched"):
                continue
            tag = (f"A{a['task_a']}_B{a['task_b']}_{a['switch_at'].replace(':', '')}"
                   f"_flush_ep{e['episode']}")
            p = Path("outputs/switch/traj") / f"{tag}.npz"
            if not p.exists():
                continue
            z = np.load(p)
            ph = z["phase"].astype(str)
            idx = np.where(ph == "B")[0]
            if len(idx) < WIN + 10:
                continue
            eps.append({"fail": 0 if e.get("b_success") else 1,
                        "pair": (a["task_a"], a["task_b"], a["switch_at"]),
                        "dither": dither_series(z["pos"][idx])})

    print(f"에피소드 {len(eps)}개 (실패 {sum(e['fail'] for e in eps)})")
    ts = list(range(WIN, MAX_T + 1, 20))
    rows = []
    for t in ts:
        sc = [e["dither"][t] if t < len(e["dither"]) else e["dither"][~np.isnan(e["dither"])][-1]
              for e in eps]
        a_, n = auc(sc, [e["fail"] for e in eps])
        med_f = np.nanmedian([s for s, e in zip(sc, eps) if e["fail"]])
        med_s = np.nanmedian([s for s, e in zip(sc, eps) if not e["fail"]])
        rows.append({"step": t, "sec": t / 20, "auc": a_, "n": n,
                     "median_fail": float(med_f), "median_success": float(med_s)})

    print(f"\n{'스텝':>6s}{'초':>6s}{'AUC':>8s}{'실패 중앙값':>12s}{'성공 중앙값':>12s}")
    for r in rows:
        print(f"{r['step']:6d}{r['sec']:6.1f}{(r['auc'] if r['auc'] else float('nan')):8.3f}"
              f"{r['median_fail']:12.2f}{r['median_success']:12.2f}")

    good = [r for r in rows if r["auc"] and r["auc"] >= 0.7]
    print()
    if good:
        r = good[0]
        print(f"→ **{r['sec']:.1f}초부터 AUC {r['auc']:.2f}** — 제자리 왕복으로 실패를 조기에 알 수 있다")
        print("   감지하면 다시 뽑거나(재추론) 동작을 흔들어 주는 방법이 성립한다")
    else:
        best = max((r for r in rows if r["auc"]), key=lambda r: r["auc"], default=None)
        if best:
            print(f"→ 최고 AUC {best['auc']:.2f} ({best['sec']:.1f}초). 0.7 을 못 넘어 조기 감지가 어렵다")

    # 쌍 안에서도 성립하는지 (쉬운 쌍/어려운 쌍 교란 제거)
    print("\n== 같은 쌍·시점 안에서의 AUC (교란 제거)")
    for t in [WIN, 80, 120, 200]:
        aucs = []
        for pair in {e["pair"] for e in eps}:
            sub = [e for e in eps if e["pair"] == pair]
            if len({e["fail"] for e in sub}) < 2:
                continue
            sc = [e["dither"][t] if t < len(e["dither"]) else e["dither"][~np.isnan(e["dither"])][-1]
                  for e in sub]
            a_, _ = auc(sc, [e["fail"] for e in sub])
            if a_ is not None:
                aucs.append(a_)
        if aucs:
            print(f"  {t / 20:4.1f}초: 집단 {len(aucs)}개 평균 AUC {np.mean(aucs):.3f} "
                  f"(0.5 초과 집단 {sum(1 for x in aucs if x > 0.5)}/{len(aucs)})")

    Path("outputs/report").mkdir(parents=True, exist_ok=True)
    json.dump(rows, open("outputs/report/dither_detector.json", "w"), ensure_ascii=False, indent=2)

    # ---- 그래프 ----
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150, facecolor=SURF)
    for ax in axs:
        ax.set_facecolor(SURF)
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(INK2); ax.tick_params(colors=INK2, length=0)
        ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    L = min(MAX_T, max(len(e["dither"]) for e in eps))
    x = np.arange(L) / 20
    for lab, key, col in [("B 실패", 1, "#eb6834"), ("B 성공", 0, "#1baf7a")]:
        M = np.full((sum(1 for e in eps if e["fail"] == key), L), np.nan)
        for i, e in enumerate([e for e in eps if e["fail"] == key]):
            d = e["dither"][:L]
            M[i, :len(d)] = d
        axs[0].plot(x, np.nanmedian(M, axis=0), color=col, lw=2, label=lab)
        axs[0].fill_between(x, np.nanpercentile(M, 25, axis=0), np.nanpercentile(M, 75, axis=0),
                            color=col, alpha=0.15, lw=0)
    axs[0].set_xlabel("전환 이후 시간 (초)", color=INK2)
    axs[0].set_ylabel("제자리 왕복 지표 (경로/범위)", color=INK2)
    axs[0].set_title("실패하면 좁은 영역에서 왕복한다", color=INK, loc="left", fontsize=11)
    axs[0].legend(frameon=False, labelcolor=INK, fontsize=9)

    axs[1].plot([r["sec"] for r in rows], [r["auc"] for r in rows], color="#2a78d6", lw=2, marker="o", ms=3)
    axs[1].axhline(0.7, color=INK2, ls="--", lw=1)
    axs[1].axhline(0.5, color=GRID, lw=1)
    axs[1].set_ylim(0.4, 1.0)
    axs[1].set_xlabel("전환 이후 시간 (초)", color=INK2)
    axs[1].set_ylabel("실패 예측 AUC", color=INK2)
    axs[1].set_title("얼마나 일찍 알 수 있나", color=INK, loc="left", fontsize=11)

    fig.suptitle("제자리 왕복으로 실패를 조기 감지할 수 있는가", x=0.01, ha="left", color=INK, fontsize=12)
    fig.text(0.01, 0.005, f"그 자리 전환(flush) {len(eps)} 에피소드. 창 {WIN}스텝(2초). AUC 0.5=무작위, 1.0=완벽",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig("outputs/report/dither_detector.png", facecolor=SURF)
    print("\n저장: outputs/report/dither_detector.png")


if __name__ == "__main__":
    main()
