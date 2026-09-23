#!/usr/bin/env python
"""
"교란된 자세에서 집기만 무너진다" 를 확정한다

배경 (2026-09-23)
  전환 뒤 팔은 목표 물체 **5cm 까지 다가가고 다시 집으려 시도도 한다**(8/10). 그런데 못 집는다.
  전환 실험에서도 B 가 물체면 1.4%, 가구·기구면 44.6% (30배).

  → 가설: **교란된 자세에서 무너지는 것은 '이동'이 아니라 '집기'다.**

무엇을 비교하나
  **집어야 하는 태스크** (1 그릇→스토브, 8 그릇→접시, 9 와인→선반)
  **안 집어도 되는 태스크** (0 서랍 열기, 7 스토브 켜기)
  를 같은 교란(위치 0·12·20·27cm)에서 돌린다.

읽는 법
  두 집단이 같이 떨어진다        → 교란이 **전반적으로** 해롭다. 가설이 틀렸다
  집는 쪽만 가파르게 떨어진다     → ✅ 가설 지지. **집기가 취약점**이다
  27cm 에서 집는 쪽만 0 이 된다   → 전환 실패가 그대로 설명된다

쓰는 법
  python analyze_regrasp.py [폴더]        (기본: outputs/regrasp)
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

GRASP = {1: "그릇→스토브", 8: "그릇→접시", 9: "와인→선반"}
NOGRASP = {0: "서랍 열기", 7: "스토브 켜기"}


def load(src):
    rows = {}
    for f in glob.glob(f"{src}/T*.json"):
        d = json.load(open(f))
        a, s = d["args"], d["summary"]
        if s.get("success_rate") is None:
            continue
        rows[(a["task"], float(a["offset_cm"]), float(a["yaw_deg"]))] = (
            s["success_rate"], s.get("n_reached", s["n"]))
    return rows


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "outputs/regrasp"
    rows = load(src)
    if not rows:
        print(f"{src} 에 결과가 없습니다. run_regrasp.sh 를 먼저 돌려야 합니다.")
        return

    offs = sorted({o for _, o, y in rows if y == 0})
    print(f"== 위치 교란별 성공률 ({src})\n")
    print(f"{'태스크':<20}" + "".join(f"{str(int(o)) + 'cm':>9}" for o in offs))

    def block(tasks, label):
        print(f"\n  [{label}]")
        curves = {}
        for t, nm in tasks.items():
            line = f"  T{t} {nm:<15}"
            c = []
            for o in offs:
                v = rows.get((t, o, 0.0))
                if v:
                    line += f"{100 * v[0]:8.0f}%"
                    c.append(v[0])
                else:
                    line += "        -"
                    c.append(np.nan)
            print(line)
            curves[t] = c
        return curves

    g = block(GRASP, "물체를 **집어야** 하는 태스크")
    ng = block(NOGRASP, "**집을 필요 없는** 태스크")

    def avg(d, i):
        v = [d[t][i] for t in d if not np.isnan(d[t][i])]
        return float(np.mean(v)) if v else float("nan")

    def fmt(v, suf="%"):
        return f"{'-':>8}" if np.isnan(v) else f"{100 * v:7.0f}{suf}"

    gm = [avg(g, i) for i in range(len(offs))]
    ngm = [avg(ng, i) for i in range(len(offs))]
    print("\n  평균")
    print(f"  {'집어야 함':<18}" + "".join(fmt(v) for v in gm))
    print(f"  {'집을 필요 없음':<17}" + "".join(fmt(v) for v in ngm))
    print(f"  {'차이':<19}" + "".join(
        f"{'-':>9}" if (np.isnan(a) or np.isnan(b)) else f"{100 * (b - a):+7.0f}%p"
        for a, b in zip(gm, ngm)))

    print("\n== 전환과 같은 교란 (위치 27cm + 회전 8도)")
    any8 = False
    for tasks, lab in [(GRASP, "집어야 함"), (NOGRASP, "집을 필요 없음")]:
        vals = []
        for t in tasks:
            v = rows.get((t, 27.0, 8.0))
            if v:
                vals.append(v[0])
                any8 = True
        if vals:
            print(f"  {lab:<16} {100 * np.mean(vals):5.0f}%   " +
                  ", ".join(f"T{t} {100 * rows[(t, 27.0, 8.0)][0]:.0f}%"
                            for t in tasks if (t, 27.0, 8.0) in rows))
    if not any8:
        print("  (아직 측정 전)")

    print("\n== 판정")
    if np.isnan(gm[-1]) or np.isnan(ngm[-1]):
        print("  아직 큰 교란 결과가 안 나왔습니다.")
        return
    far_gap = ngm[-1] - gm[-1]
    near_gap = ngm[0] - gm[0]
    print(f"  교란 없음({offs[0]:.0f}cm): 집어야 함 {100 * gm[0]:.0f}%  vs  집을 필요 없음 {100 * ngm[0]:.0f}%"
          f"   (차이 {100 * near_gap:+.0f}%p)")
    print(f"  교란 {offs[-1]:.0f}cm     : 집어야 함 {100 * gm[-1]:.0f}%  vs  집을 필요 없음 {100 * ngm[-1]:.0f}%"
          f"   (차이 {100 * far_gap:+.0f}%p)")
    if far_gap - near_gap > 0.2:
        print("\n  ✅ **가설 지지.** 교란이 커질수록 **집는 쪽만** 더 크게 무너진다.")
        print("     → 전환 실패(B 가 물체면 1.4%, 가구·기구면 44.6%)가 그대로 설명된다")
        print("     → 고쳐야 할 것은 '전환'이 아니라 **교란된 자세에서 집는 능력**이다")
    elif abs(far_gap - near_gap) <= 0.2:
        print("\n  ❌ **가설과 다르다.** 두 집단이 비슷하게 떨어진다.")
        print("     → 교란은 전반적으로 해롭고, '집기'가 특별히 취약한 것은 아니다")
        print("     → B 효과(1.4% vs 44.6%)는 다른 이유를 찾아야 한다")
    else:
        print("\n  ⚠️ 오히려 **안 집는 쪽**이 더 크게 무너진다. 예상과 반대다")

    # 데이터 수집 가능 범위 — LoRA 4차의 단계 설계에 쓴다
    print("\n== 학습 데이터를 모을 수 있는 범위 (성공률 5% 이상이어야 다시 뽑기가 통한다)")
    for t in GRASP:
        ok = [offs[i] for i in range(len(offs)) if not np.isnan(g[t][i]) and g[t][i] >= 0.05]
        zero = [offs[i] for i in range(len(offs)) if not np.isnan(g[t][i]) and g[t][i] < 0.05]
        print(f"  T{t}: 모을 수 있음 ~{max(ok):.0f}cm" if ok else f"  T{t}: 어디서도 못 모음", end="")
        print(f"   / 사실상 0% : {min(zero):.0f}cm 이상" if zero else "")
    print("  → 0% 구간은 **단계적 확장**으로만 닿을 수 있다 (낮은 교란에서 먼저 학습)")


if __name__ == "__main__":
    main()
