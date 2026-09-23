#!/usr/bin/env python
"""
그리퍼 유지(grip latch) 평가 — 놓지 못하게 하면 무엇이 달라지나

배경 (2026-09-23)
  "그릇을 서랍에 넣어라" 를 **그릇을 쥔 채** 받아도 정책은 10/10 그릇을 놓는다.
  '지시가 바뀌면 일단 놓는다' 가 학습된 반사다.
  B 의 목표가 물체인 쌍은 전환 성공 1.4%, 가구·기구인 쌍은 44.6% (30배).

무엇을 보나
  ① **B 성공** — 쥔 채로 새 일을 할 수 있는가
  ② ⭐ **A 재개** — 지금까지 **0/62 로 완전히 막혀 있던** 두 번째 목표.
     물체를 놓지 않았으면 원래 하던 일로 돌아갈 수 있어야 한다
  ③ **해롭지 않은가** — 물체가 필요 없는 B(서랍·스토브)에서 성능이 떨어지면 안 된다
  ④ 자연스러움(jerk) — 그리퍼만 덮어쓰므로 팔 궤적은 그대로여야 한다

읽는 법
  B ↑        → 쥔 채로 새 일을 할 수 있다
  A 재개 ↑   → **연구의 두 번째 목표가 처음으로 움직인다**
  B 유지, A 재개 ↑ → 전환은 못 고쳐도 재개는 고친 것. 그것만으로도 결과다
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

BASE = "outputs/switch"
LATCH = "outputs/latch"


def load(path):
    if not Path(path).exists():
        return None
    return [e for e in json.load(open(path))["episodes"] if e.get("switched")]


def main():
    rows = {}
    for f in glob.glob(f"{LATCH}/A*_grasp3_flush_gl*.json"):
        m = re.match(r"A(\d+)_B(\d+)_grasp3_flush_gl(\d+)$", Path(f).stem)
        if not m:
            continue
        a, b, g = int(m.group(1)), int(m.group(2)), int(m.group(3))
        eps = load(f)
        if eps:
            rows[(a, b, g)] = eps
    if not rows:
        print(f"{LATCH} 에 결과가 없습니다. run_latch.sh 를 먼저 돌려야 합니다.")
        return

    pairs = sorted({(a, b) for a, b, _ in rows})
    gs = sorted({g for _, _, g in rows})

    def cell(eps, key):
        return sum(bool(e.get(key)) for e in eps), len(eps)

    for key, title in [("b_success", "① B 성공률"),
                       ("a_resume_success", "② ⭐ A 재개 성공률 (지금까지 0/62)")]:
        print(f"\n== {title}\n")
        print(f"{'쌍':<8}{'기준':>9}" + "".join(f"{'유지 ' + str(g):>10}" for g in gs))
        tb = [0, 0]
        tl = {g: [0, 0] for g in gs}
        for a, b in pairs:
            base = load(f"{BASE}/A{a}_B{b}_grasp3_flush.json")
            line = f"A{a}→B{b:<3}"
            if base:
                k, n = cell(base, key)
                tb[0] += k; tb[1] += n
                line += f"{100 * k / n:8.0f}%"
            else:
                line += "        -"
            for g in gs:
                eps = rows.get((a, b, g))
                if eps:
                    k, n = cell(eps, key)
                    tl[g][0] += k; tl[g][1] += n
                    line += f"{100 * k / n:9.0f}%"
                else:
                    line += "         -"
            print(line)
        line = f"{'합계':<8}{100 * tb[0] / max(tb[1], 1):8.0f}%"
        for g in gs:
            line += f"{100 * tl[g][0] / max(tl[g][1], 1):9.0f}%"
        print(line)
        print(f"{'':8}{f'({tb[0]}/{tb[1]})':>9}"
              + "".join(f"{f'({tl[g][0]}/{tl[g][1]})':>10}" for g in gs))

        # 같은 쌍만 모아 검정 (유지 조건이 있는 쌍으로 한정해야 공정하다)
        for g in gs:
            kb = nb = kl = nl = 0
            for a, b in pairs:
                eps = rows.get((a, b, g))
                base = load(f"{BASE}/A{a}_B{b}_grasp3_flush.json")
                if not eps or not base:
                    continue
                k, n = cell(base, key); kb += k; nb += n
                k, n = cell(eps, key); kl += k; nl += n
            if nl and nb:
                p = fisher_exact([[kl, nl - kl], [kb, nb - kb]], alternative="greater")[1]
                mark = "✅" if p < 0.05 else ""
                print(f"  유지 {g:3d}: {kl}/{nl} vs 기준 {kb}/{nb}   p = {p:.4f} {mark}")

    print("\n== ④ 자연스러움 (jerk 중앙값) — 그리퍼만 덮어쓰므로 팔 궤적은 그대로여야 한다")
    jb, jl = [], {g: [] for g in gs}
    for a, b in pairs:
        base = load(f"{BASE}/A{a}_B{b}_grasp3_flush.json")
        if base:
            jb += [e["jerk"] for e in base if e.get("jerk") is not None]
        for g in gs:
            eps = rows.get((a, b, g))
            if eps:
                jl[g] += [e["jerk"] for e in eps if e.get("jerk") is not None]
    if jb:
        print(f"  기준 {np.median(jb):.2f}", end="")
        for g in gs:
            if jl[g]:
                r = np.median(jl[g]) / np.median(jb)
                print(f"   유지 {g}: {np.median(jl[g]):.2f} ({r:.2f}배)", end="")
        print()

    print("\n== 판정")
    for key, lbl in [("b_success", "B 성공"), ("a_resume_success", "A 재개")]:
        best = None
        for g in gs:
            kb = nb = kl = nl = 0
            for a, b in pairs:
                eps = rows.get((a, b, g))
                base = load(f"{BASE}/A{a}_B{b}_grasp3_flush.json")
                if not eps or not base:
                    continue
                k, n = cell(base, key); kb += k; nb += n
                k, n = cell(eps, key); kl += k; nl += n
            if nl and (best is None or kl / nl > best[1]):
                best = (g, kl / nl, kb / max(nb, 1), kl, nl, kb, nb)
        if best:
            g, v, bv, kl, nl, kb, nb = best
            d = 100 * (v - bv)
            print(f"  {lbl:<7} 기준 {100 * bv:3.0f}% → 최고(유지 {g}) {100 * v:3.0f}%  ({d:+.0f}%p)")
    print("\n  ⚠️ B 가 안 올라도 A 재개가 오르면 그것만으로 결과다 —")
    print("     연구의 두 번째 목표(중단한 일로 돌아오기)가 지금까지 0% 였다.")


if __name__ == "__main__":
    main()
