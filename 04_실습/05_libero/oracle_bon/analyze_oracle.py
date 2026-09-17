#!/usr/bin/env python
"""
오라클 Best-of-N 분석 — "후보 중 고르기"에 남은 여지가 있는가?

읽는 법
  기본 성공률 p    후보 하나를 그냥 실행 (= flush)
  오라클           N개 중 하나라도 성공한 상태의 비율 → **모든 후보 선택 방법의 상한**
  독립 예측        1-(1-p)^N. 성공이 순전히 생성 노이즈 운이라면 오라클이 여기에 가까워야 함
  과분산           관측 분산 ÷ 이항분포 분산. 1 이면 순전히 운, 크면 상태가 결정
  전부 실패한 상태  여기서는 어떤 후보를 골라도 안 됨 → 선택으로 못 고치는 부분
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["NanumBarunGothic", "NanumGothic", "DejaVu Sans"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"
C_BASE, C_ORACLE, C_PRED = "#8a8984", "#1baf7a", "#2a78d6"


def stats(counts, N):
    c = np.asarray(counts, dtype=float)
    if len(c) == 0:
        return None
    p = float(c.sum() / (len(c) * N))
    var_bin = N * p * (1 - p)
    return {"states": len(c), "p_base": p, "oracle": float((c > 0).mean()),
            "indep": 1 - (1 - p) ** N, "never": float((c == 0).mean()),
            "always": float((c == N).mean()),
            "overdispersion": float(c.var() / var_bin) if var_bin > 0 else None}


def main():
    d = json.load(open("outputs/oracle_bon/oracle.json"))
    N = d["args"]["n"]
    rows = d["rows"]
    groups = {}
    for r in rows:
        groups.setdefault(f"{r['task_a']}→{r['task_b']}", []).append(r["n_success"])
    allc = [r["n_success"] for r in rows]

    out = {"N": N, "overall": stats(allc, N), "by_pair": {k: stats(v, N) for k, v in groups.items()}}
    o = out["overall"]
    print(f"== 오라클 Best-of-N (상태 {o['states']}개 × 후보 {N}개, {d['args']['switch_at']})\n")
    print(f"{'조건':10s}{'상태':>5s}{'기본 p':>9s}{'오라클':>9s}{'독립예측':>9s}{'전부실패':>9s}{'과분산':>8s}")
    for k, s in out["by_pair"].items():
        print(f"{k:10s}{s['states']:5d}{100 * s['p_base']:8.0f}%{100 * s['oracle']:8.0f}%"
              f"{100 * s['indep']:8.0f}%{100 * s['never']:8.0f}%{s['overdispersion']:8.2f}")
    print(f"{'합계':10s}{o['states']:5d}{100 * o['p_base']:8.0f}%{100 * o['oracle']:8.0f}%"
          f"{100 * o['indep']:8.0f}%{100 * o['never']:8.0f}%{o['overdispersion']:8.2f}")

    print("\n성공 개수 분포 (0~N 중 몇 개 성공했나)")
    hist = np.bincount(allc, minlength=N + 1)
    for i, h in enumerate(hist):
        print(f"  {i}/{N}: {'█' * h} {h}")

    # 해석
    o2 = out["overall"]
    print("\n== 해석")
    gain = o2["oracle"] - o2["p_base"]
    print(f"완벽한 선택기가 있으면 {100 * o2['p_base']:.0f}% → {100 * o2['oracle']:.0f}% "
          f"(+{100 * gain:.0f}%p) 까지 오를 수 있다")
    if o2["overdispersion"] is not None and o2["overdispersion"] > 1.5:
        print(f"과분산 {o2['overdispersion']:.2f} > 1.5 → 성공/실패가 **상태에서 갈린다**. "
              f"전부 실패한 상태 {100 * o2['never']:.0f}% 는 선택으로 못 고친다")
    else:
        print(f"과분산 {o2['overdispersion']:.2f} → 결과가 상당 부분 **생성 노이즈 운**. 선택이 통할 여지가 있다")

    # ---- 시드(노이즈) 효과: 모든 상태에 같은 8개 시드를 썼으므로 비교 가능 ----
    S = np.array([r["success"] for r in rows], dtype=float)  # (상태, 시드)
    per_seed = S.mean(axis=0)
    out["per_seed"] = [float(x) for x in per_seed]
    print("\n== 노이즈 시드별 성공률 (모든 상태에 같은 시드 8개를 사용)")
    for k, v in enumerate(per_seed):
        print(f"  시드 {k}: {'█' * int(round(20 * v))} {100 * v:.0f}%")
    # 순열 검정: 상태 안에서 시드 라벨을 섞어도 이만큼 갈리는가
    rng = np.random.default_rng(0)
    obs_spread = per_seed.max() - per_seed.min()
    null = np.empty(2000)
    for b in range(len(null)):
        P = np.apply_along_axis(rng.permutation, 1, S)
        m = P.mean(axis=0)
        null[b] = m.max() - m.min()
    pval = float((null >= obs_spread).mean())
    out["seed_effect"] = {"spread": float(obs_spread), "p": pval,
                          "best_seed": int(per_seed.argmax()), "best_rate": float(per_seed.max())}
    print(f"시드 간 최대-최소 차이 {100 * obs_spread:.0f}%p, 순열 검정 p = {pval:.3f}")
    if pval < 0.05:
        print(f"→ **어떤 노이즈는 체계적으로 더 좋다.** 가장 좋은 시드 {per_seed.argmax()} 가 "
              f"{100 * per_seed.max():.0f}% (평균 {100 * o2['p_base']:.0f}%)")
        print("   학습 없이 '좋은 노이즈'를 고르는 것만으로 개선 가능 → DSRL 의 전제")
    else:
        print("→ 특정 노이즈가 더 좋다는 증거 없음. 상태마다 좋은 노이즈가 다르다는 뜻")

    # ---- 그래프 ----
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=150, facecolor=SURF,
                            gridspec_kw={"width_ratios": [1.5, 1]})
    for ax in axs:
        ax.set_facecolor(SURF)
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(INK2); ax.tick_params(colors=INK2, length=0)
        ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)

    keys = list(out["by_pair"]) + ["합계"]
    vals = [out["by_pair"][k] for k in out["by_pair"]] + [o]
    x = np.arange(len(keys)); w = 0.27
    for j, (lab, key, col) in enumerate([("기본 (후보 1개)", "p_base", C_BASE),
                                         ("오라클 (8개 중 최선)", "oracle", C_ORACLE),
                                         ("독립 가정 예측", "indep", C_PRED)]):
        axs[0].bar(x + (j - 1) * (w + 0.01), [100 * v[key] for v in vals], w, color=col, label=lab, zorder=2)
    axs[0].set_xticks(x, keys, fontsize=9, color=INK); axs[0].set_ylim(0, 105)
    axs[0].set_ylabel("B 성공률 (%)", color=INK2)
    axs[0].set_title("후보를 완벽하게 고를 수 있다면 어디까지 오르나", color=INK, loc="left", fontsize=11)
    axs[0].legend(frameon=False, labelcolor=INK, fontsize=8)

    axs[1].bar(np.arange(N + 1), hist, color="#2a78d6", zorder=2)
    axs[1].set_xticks(np.arange(N + 1), [str(i) for i in range(N + 1)], fontsize=9, color=INK)
    axs[1].set_xlabel(f"{N}개 후보 중 성공한 개수", color=INK2)
    axs[1].set_ylabel("상태 수", color=INK2)
    axs[1].set_title("같은 상태에서 다시 굴렸을 때", color=INK, loc="left", fontsize=11)

    fig.suptitle("오라클 Best-of-N — 같은 상태·같은 지시, 생성 노이즈만 다르게 8번",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.text(0.01, 0.005, f"{o['states']}개 전환 상태 × 후보 {N}개 = {o['states'] * N} 회 rollout. "
                          f"시뮬레이터 상태를 저장/복원해 같은 지점에서 다시 시작", color=INK2, fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    Path("outputs/report").mkdir(parents=True, exist_ok=True)
    fig.savefig("outputs/report/oracle_bon.png", facecolor=SURF)
    json.dump(out, open("outputs/report/oracle_bon.json", "w"), ensure_ascii=False, indent=2)
    print("\n저장: outputs/report/oracle_bon.png, oracle_bon.json")


if __name__ == "__main__":
    main()
