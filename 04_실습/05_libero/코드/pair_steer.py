#!/usr/bin/env python
"""
쌍별 조종 가능성이 전환 성공률을 예측하는가

왜 필요한가
  지금까지 모든 방법이 **쉬운 쌍에서만 조금 움직이고 어려운 쌍은 0%** 였다.

      8→7  50%      4→7  30%      1→5  0%      2→5  17%

  왜 갈리는지 모른다. 오라클(같은 상태에서 16번 다시 뽑기)로도
  어려운 쌍은 0~17% 라, **애초에 좋은 후보가 없다.**

  전체 CMI 는 지시문 10개를 뭉뚱그린 값이라 쌍별 난이도를 못 본다.
  그래서 **쌍별 분리도**를 쟀다:

      분리도(A→B) = ‖(B 지시로 낸 동작의 평균) − (A 지시로 낸 동작의 평균)‖
                    ────────────────────────────────────────────────────
                          같은 지시 안의 흔들림 (생성 노이즈)

  1 에 가까우면 **지시를 B 로 바꿔도 노이즈만큼도 안 달라진다** = 그 쌍은 가망이 없다.

읽는 법
  분리도와 성공률이 같이 간다  →  ✅ **쌍 난이도의 원인이 조종 가능성이다.**
                                 새 쌍을 굴려보지 않고도 난이도를 예측할 수 있다
  관계가 없다                  →  ❌ 난이도는 다른 데서 온다 (물체 배치, 경로 충돌 등)

쓰는 법
  python pair_steer.py [steer 폴더]     (기본: outputs/steer_curve)
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SWITCH_DIR = "outputs/switch"
PROBE = "grasp3"  # 전환 실험과 같은 시점


def switch_rates():
    """실제 전환 성공률 {(A, B): (성공, 전체)} — 원본 flush, 잡은 직후."""
    out = {}
    for f in glob.glob(f"{SWITCH_DIR}/A*_B*_{PROBE}_flush.json"):
        stem = Path(f).stem
        head = stem.split("_")[0:2]
        try:
            a, b = int(head[0][1:]), int(head[1][1:])
        except ValueError:
            continue
        eps = [e for e in json.load(open(f))["episodes"] if e.get("switched")]
        if eps:
            out[(a, b)] = (sum(bool(e.get("b_success")) for e in eps), len(eps))
    return out


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "outputs/steer_curve"
    f = Path(src) / "steer.json"
    if not f.exists():
        print(f"{f} 가 없습니다. steerability.py 를 먼저 돌려야 합니다.")
        return
    rows = [r for r in json.load(open(f))["rows"]
            if r["probe"] == PROBE and r.get("pair_sep")]
    if not rows:
        print(f"'{PROBE}' 시점의 쌍별 분리도가 없습니다.")
        print("(2026-09-21 이후 steerability.py 로 --exclude-own 없이 다시 재야 합니다)")
        return

    # (A, B) 별 분리도 평균
    sep = {}
    for r in rows:
        a = r["task_a"]
        for b_str, v in r["pair_sep"].items():
            sep.setdefault((a, int(b_str)), []).append(v)
    sep = {k: float(np.mean(v)) for k, v in sep.items()}

    rates = switch_rates()
    common = sorted(set(sep) & set(rates))
    print(f"== 쌍별 조종 가능성 vs 실제 전환 성공률 ({src}, {PROBE})\n")
    if not common:
        print("전환 실험 결과가 있는 쌍이 없습니다. 아래는 분리도만 보여 줍니다.\n")
        for (a, b), v in sorted(sep.items(), key=lambda x: -x[1])[:20]:
            print(f"  {a}→{b}  분리도 {v:6.2f}")
        return

    print(f"{'쌍':>7}{'분리도':>9}{'전환 성공':>12}")
    xs, ys = [], []
    for a, b in common:
        k, n = rates[(a, b)]
        print(f"  {a}→{b}{sep[(a, b)]:9.2f}{100 * k / n:10.0f}%  ({k}/{n})")
        xs.append(sep[(a, b)]); ys.append(k / n)

    if len(common) >= 4:
        rho, p = spearmanr(xs, ys)
        print(f"\n스피어만 상관 ρ = {rho:+.3f}  (p = {p:.3f}, 쌍 {len(common)}개)")
        if rho > 0.5 and p < 0.1:
            print("→ ✅ **분리도가 큰 쌍일수록 전환이 잘 된다.**")
            print("   쌍 난이도의 원인이 조종 가능성이라는 뜻이다.")
            print("   → 굴려보지 않고 **추론만으로 새 쌍의 난이도를 예측**할 수 있다")
        elif abs(rho) < 0.3:
            print("→ ❌ 관계가 없다. 쌍 난이도는 조종 가능성으로 설명되지 않는다")
            print("   → 다른 후보: 물체 배치, B 경로가 A 가 놓은 물체와 겹침, 목표 판정의 엄격함")
        else:
            print("→ ⚠️ 관계가 약하다. 쌍이 너무 적어 판단하기 이르다")
    else:
        print(f"\n쌍이 {len(common)}개뿐이라 상관을 보기 어렵습니다. 전환 실험을 더 넓혀야 합니다.")

    print("\n== 가장 조종하기 어려운 쌍 (분리도 낮은 순, 전환 실험 없는 것 포함)")
    for (a, b), v in sorted(sep.items(), key=lambda x: x[1])[:8]:
        mark = ""
        if (a, b) in rates:
            k, n = rates[(a, b)]
            mark = f"   (실측 {100 * k / n:.0f}%)"
        print(f"  {a}→{b}  분리도 {v:6.2f}{mark}")
    print("\n분리도가 1 근처면 '지시를 바꿔도 생성 노이즈만큼도 안 달라진다'는 뜻이다")


if __name__ == "__main__":
    main()
