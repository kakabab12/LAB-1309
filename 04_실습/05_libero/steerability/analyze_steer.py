#!/usr/bin/env python
"""조종 가능성(CMI) 측정 결과 정리 + 그래프.

가설: **물체를 쥐고 있으면 지시문이 동작에 영향을 못 준다** (instruction-blind)
확인법: 측정 지점별 CMI 를 비교. 시작 > 잡은 직후 > 들고 이동 중 순으로 낮아지면 가설 지지
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["NanumBarunGothic", "NanumGothic", "DejaVu Sans"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"
LABEL = {"start": "에피소드 시작", "grasp3": "잡은 직후", "grasp20": "들고 이동 중"}
COL = {"start": "#1baf7a", "grasp3": "#eb6834", "grasp20": "#b8452c"}


def main():
    d = json.load(open("outputs/steerability/steer.json"))
    rows = d["rows"]
    probes = [p for p in ["start", "grasp3", "grasp20"] if any(r["probe"] == p for r in rows)]
    tasks = sorted({r["task_a"] for r in rows})

    def sel(probe, task=None):
        return [r for r in rows if r["probe"] == probe and (task is None or r["task_a"] == task)]

    print("== 조종 가능성 (CMI, 클수록 지시가 동작을 가른다)\n")
    print(f"{'측정 지점':14s}{'n':>4s}{'CMI(nats)':>11s}{'분리비':>9s}{'자기-타지시 거리':>16s}")
    summary = {}
    for p in probes:
        s = sel(p)
        summary[p] = {"n": len(s),
                      "cmi": float(np.mean([r["cmi_nats"] for r in s])),
                      "cmi_sd": float(np.std([r["cmi_nats"] for r in s])),
                      "sep": float(np.mean([r["sep_ratio"] for r in s])),
                      "dist": float(np.mean([r["dist_own_vs_others"] for r in s]))}
        v = summary[p]
        print(f"{LABEL[p]:14s}{v['n']:4d}{v['cmi']:11.4f}{v['sep']:9.2f}{v['dist']:16.3f}")

    if "start" in summary and "grasp3" in summary:
        a, b = summary["start"]["cmi"], summary["grasp3"]["cmi"]
        print(f"\n시작 → 잡은 직후: CMI {a:.4f} → {b:.4f} ({100 * (b - a) / max(a, 1e-9):+.0f}%)")
        if b < a * 0.8:
            print("→ 물체를 쥐면 지시문의 영향력이 확실히 줄어든다 (instruction-blind 가설 지지)")
        elif b > a * 1.2:
            print("→ 오히려 늘었다. 지시는 들리는데 **수행을 못 하는** 문제 (9/17 결론과 일치)")
        else:
            print("→ 큰 차이 없음. 지시 이해 자체는 유지된다는 뜻")

    print("\n태스크별 CMI")
    print(f"{'태스크':8s}" + "".join(f"{LABEL[p]:>14s}" for p in probes))
    for t in tasks:
        line = f"T{t:<7d}"
        for p in probes:
            s = sel(p, t)
            line += f"{np.mean([r['cmi_nats'] for r in s]):14.4f}" if s else f"{'-':>14s}"
        print(line)

    # ---- 그래프 ----
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150, facecolor=SURF)
    for ax in axs:
        ax.set_facecolor(SURF)
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(INK2); ax.tick_params(colors=INK2, length=0)
        ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)

    xs = np.arange(len(probes))
    axs[0].bar(xs, [summary[p]["cmi"] for p in probes],
               yerr=[summary[p]["cmi_sd"] for p in probes], capsize=4,
               color=[COL[p] for p in probes], zorder=2, error_kw={"ecolor": INK2, "lw": 1})
    axs[0].set_xticks(xs, [LABEL[p] for p in probes], fontsize=9, color=INK)
    axs[0].set_ylabel("CMI (nats)", color=INK2)
    axs[0].set_title("지시문이 동작을 얼마나 가르는가", color=INK, loc="left", fontsize=11)

    w = 0.8 / max(len(probes), 1)
    xt = np.arange(len(tasks))
    for j, p in enumerate(probes):
        ys = [np.mean([r["cmi_nats"] for r in sel(p, t)]) if sel(p, t) else np.nan for t in tasks]
        axs[1].bar(xt + (j - (len(probes) - 1) / 2) * w, ys, w * 0.92, color=COL[p],
                   label=LABEL[p], zorder=2)
    axs[1].set_xticks(xt, [f"태스크 {t}" for t in tasks], fontsize=9, color=INK)
    axs[1].set_title("태스크별", color=INK, loc="left", fontsize=11)
    axs[1].legend(frameon=False, labelcolor=INK, fontsize=8)

    fig.suptitle("조종 가능성 — 같은 상태에서 지시문 10개 × 후보 8개를 뽑아 비교 (ReSteer 의 CMI)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.text(0.01, 0.005, "CMI = ½·평균 log(전체 분산 / 같은 지시 안의 분산). 0 이면 지시가 동작에 영향 없음",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    Path("outputs/report").mkdir(parents=True, exist_ok=True)
    fig.savefig("outputs/report/steerability.png", facecolor=SURF)
    json.dump({"summary": summary, "by_task": {str(t): {p: (float(np.mean([r["cmi_nats"] for r in sel(p, t)]))
                                                            if sel(p, t) else None) for p in probes}
                                               for t in tasks}},
              open("outputs/report/steerability.json", "w"), ensure_ascii=False, indent=2)
    print("\n저장: outputs/report/steerability.png, steerability.json")


if __name__ == "__main__":
    main()
