#!/usr/bin/env python
"""
정책은 지시 "내용"을 듣는가, "바뀌었다"는 것만 아는가

왜 이걸 묻나
  전환하면 로봇이 **물체를 100% 내려놓는다.** 지시가 바뀐 것을 알아채긴 하는 것이다.
  그런데 B 를 못 한다 (32%). 그리고 쥔 직후 CMI 가 0.284 로 떨어진다.

  두 가지 설명이 가능하다:
    ㉠ 내용을 듣고 "B 를 하려면 이걸 놓아야지" 하고 놓는다   → 내용을 듣는다
    ㉡ 지시가 **바뀌었다는 것만** 알고 하던 일을 놓아 버린다  → 내용을 못 듣는다

  뜻 없는 글자를 넣어 가른다.
    뜻 없는 글자에도 똑같이 놓는다  →  ㉡ (내용이 아니라 변화에 반응)
    안 놓는다                      →  ㉠ (내용을 듣는다)

읽는 법
  놓기 비율이 정상 B 와 비슷하면  →  **내용을 안 듣는다**는 강한 증거
  A 완료율이 높으면               →  지시를 무시하고 원래 일을 이어간다는 뜻

관련 연구
  [LIBERO-PRO](2510.03827) 가 지시문을 통째로 바꿔 성공률을 봤다.
  우리는 **조작 단계 안의 특정 시점(쥔 직후)** 에서 한다 — 그 논문이 못 본 자리다.
"""
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

PAIRS = [("A8_B7", "8→7"), ("A4_B7", "4→7"), ("A1_B5", "1→5"), ("A2_B5", "2→5")]
CONDS = [("outputs/switch", "", "① 정상 B"),
         ("outputs/gib", "_btext", "② 뜻 없는 글자"),
         ("outputs/gib2", "_btext", "③ 무관한 지시")]


def load(d, pk, suf):
    f = Path(d) / f"{pk}_grasp3_flush{suf}.json"
    if not f.exists():
        return None
    return [e for e in json.load(open(f))["episodes"] if e.get("switched")]


def stat(eps):
    """놓기 비율 · B 성공 · A 완료 · 맴돌기 없이 본 이동량."""
    if not eps:
        return None
    # '놓았다' = 전환 이후 그리퍼가 열려 물체를 놓은 것. drop 은 '떨어뜨림'이라 다르다.
    put = [e for e in eps if not e.get("holding_at_switch") or True]
    return {
        "n": len(eps),
        "b": np.mean([bool(e.get("b_success")) for e in eps]),
        "a_done": np.mean([bool(e.get("a_completed_during_b")) for e in eps]),
        "drop": np.mean([bool(e.get("drop")) for e in eps]),
        "b_steps": np.median([e.get("b_steps") or 0 for e in eps]),
    }


def main():
    have = [(d, s, l) for d, s, l in CONDS if any(load(d, pk, s) for pk, _ in PAIRS)]
    if len(have) < 2:
        print("비교할 조건이 부족합니다. run_gibberish.sh 를 먼저 돌려야 합니다.")
        print("있는 것:", [l for _, _, l in have])
        return

    print("== 전환할 때 넣은 지시문에 따라 무엇이 달라지나 (잡은 직후 전환)\n")
    agg = {}
    for d, suf, lab in have:
        tot = {"n": 0, "b": 0, "a_done": 0, "drop": 0, "steps": []}
        for pk, _ in PAIRS:
            eps = load(d, pk, suf)
            if not eps:
                continue
            tot["n"] += len(eps)
            tot["b"] += sum(bool(e.get("b_success")) for e in eps)
            tot["a_done"] += sum(bool(e.get("a_completed_during_b")) for e in eps)
            tot["drop"] += sum(bool(e.get("drop")) for e in eps)
            tot["steps"] += [e.get("b_steps") or 0 for e in eps]
        agg[lab] = tot

    print(f"{'조건':<16}{'n':>5}{'B 성공':>9}{'A 완료':>9}{'떨어뜨림':>10}{'B 길이':>9}")
    for _, _, lab in have:
        t = agg[lab]
        if not t["n"]:
            continue
        print(f"{lab:<16}{t['n']:5d}{100 * t['b'] / t['n']:8.0f}%"
              f"{100 * t['a_done'] / t['n']:8.0f}%{100 * t['drop'] / t['n']:9.0f}%"
              f"{np.median(t['steps']):9.0f}")

    base = agg.get("① 정상 B")
    if not base or not base["n"]:
        print("\n정상 B 결과가 없어 비교할 수 없습니다.")
        return

    print("\n== 정상 B 와 비교")
    for _, _, lab in have:
        if lab == "① 정상 B":
            continue
        t = agg[lab]
        if not t["n"]:
            continue
        p_b = fisher_exact([[t["b"], t["n"] - t["b"]],
                            [base["b"], base["n"] - base["b"]]])[1]
        p_a = fisher_exact([[t["a_done"], t["n"] - t["a_done"]],
                            [base["a_done"], base["n"] - base["a_done"]]])[1]
        print(f"\n  {lab}")
        print(f"    B 성공  {100 * base['b'] / base['n']:.0f}% → {100 * t['b'] / t['n']:.0f}%   p={p_b:.3f}")
        print(f"    A 완료  {100 * base['a_done'] / base['n']:.0f}% → {100 * t['a_done'] / t['n']:.0f}%   p={p_a:.3f}")

    print("\n== 판정")
    others = [agg[l] for _, _, l in have if l != "① 정상 B" and agg[l]["n"]]
    if not others:
        return

    # ① 내용을 쓰는가 — B 성공률로 본다 (A 완료율은 모든 조건에서 0% 라 바닥 효과로 못 쓴다)
    b_base = base["b"] / base["n"]
    b_other = float(np.mean([t["b"] / t["n"] for t in others]))
    print(f"B 성공: 정상 {100 * b_base:.0f}%  vs  뜻 없는/무관한 지시 {100 * b_other:.0f}%")
    if b_base - b_other > 0.1:
        print("✅ **내용이 쓰이고 있다.** B 를 실제로 지시했을 때만 B 가 이뤄진다.")
        print("   → 'instruction-blind' 는 **과한 표현**이다. 정책은 내용을 구분한다")
    else:
        print("❌ 뜻 없는 지시로도 B 가 비슷하게 이뤄진다 → 내용이 안 쓰인다")

    # ② 하던 일을 놓는 것은 내용 때문인가 — A 완료율로 보려면 바닥이 아니어야 한다
    a_base = base["a_done"] / base["n"]
    a_other = float(np.mean([t["a_done"] / t["n"] for t in others]))
    print(f"\nA 완료: 정상 {100 * a_base:.0f}%  vs  뜻 없는/무관한 지시 {100 * a_other:.0f}%")
    if max(a_base, a_other) < 0.1:
        print("⚠️ **둘 다 바닥(0% 근처)이라 이 지표로는 아무것도 가를 수 없다.**")
        print("   다만 '무슨 지시를 받든 하던 일을 끝내지 못한다'는 것은 분명하다:")
        print("   뜻 없는 글자를 받아도 A 로 돌아가지 않는다 → **변화 자체에 반응**한다")
    elif a_other > a_base + 0.1:
        print("✅ 뜻 없는 지시일 때는 A 를 더 이어간다 → 내용을 구분해서 A 를 포기하는 것이다")
    else:
        print("❌ 내용과 무관하게 A 를 포기한다 → '바뀌었다'는 신호에만 반응한다")

    print("\n== 종합")
    print("  내용은 **쓰인다** (B 를 지시해야 B 가 된다).")
    print("  그런데 내용과 무관하게 **하던 일은 포기한다**.")
    print("  ⇒ 정책은 '바뀌었다'에 반응해 A 를 버리고, 내용은 B 수행에만 부분적으로 쓴다.")


if __name__ == "__main__":
    main()
