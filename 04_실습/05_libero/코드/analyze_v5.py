#!/usr/bin/env python
"""
LoRA 5차 판정 — 원래 정책과 **같은 에피소드(20~29)에서** 나란히 비교

읽는 순서 (앞이 무너지면 뒤는 볼 필요 없다)
  ① **망각** — 10개 태스크를 교란 없이. 4차는 여기서 무너졌다 (T8 90% → 10%)
  ② **원래 실패하던 전환** — 다른 조작이 필요한 쌍 (접시 밀기·와인병·서랍+그릇·스토브)
  ③ **원래 잘 되던 전환** — 목적지만 바뀌는 쌍 + 서랍 열기. 나빠지면 안 된다

짝 비교
  같은 에피소드 번호끼리 짝지어 "한쪽만 성공"을 센다. 초기 배치가 같으므로
  독립 비교보다 훨씬 민감하다.
"""
import glob
import json
from math import comb
from pathlib import Path

import numpy as np


def sign_test(w, l):
    k = w + l
    return min(1.0, 2 * sum(comb(k, i) for i in range(max(w, l), k + 1)) / 2 ** k) if k else 1.0


def load(d, key):
    out = {}
    for f in glob.glob(f"{d}/*.json"):
        js = json.load(open(f))
        eps = js.get("episodes", [])
        a = js.get("args", {})
        tag = Path(f).stem
        out[tag] = {e["episode"]: bool(e.get(key)) for e in eps
                    if key != "b_success" or e.get("switched")}
    return out


def compare(title, base, v5, note=""):
    print(f"\n== {title}")
    if note:
        print(f"   {note}")
    print(f"{'조건':<34}{'원래':>8}{'5차':>8}{'변화':>8}{'짝(5차만/원래만)':>18}")
    tb = tv = nb = nv = W = L = 0
    for tag in sorted(set(base) | set(v5)):
        b, v = base.get(tag, {}), v5.get(tag, {})
        if not b or not v:
            continue
        kb, kv = sum(b.values()), sum(v.values())
        common = set(b) & set(v)
        w = sum(v[e] and not b[e] for e in common)
        l = sum(b[e] and not v[e] for e in common)
        W += w; L += l
        tb += kb; nb += len(b); tv += kv; nv += len(v)
        print(f"{tag:<34}{100 * kb / len(b):7.0f}%{100 * kv / len(v):7.0f}%"
              f"{100 * (kv / len(v) - kb / len(b)):+7.0f}%p{w:9d} / {l:<3d}")
    if nb and nv:
        p = sign_test(W, L)
        print(f"{'합계':<34}{100 * tb / nb:7.0f}%{100 * tv / nv:7.0f}%"
              f"{100 * (tv / nv - tb / nb):+7.0f}%p{W:9d} / {L:<3d}   p = {p:.4f}")
        return tb / nb, tv / nv, p, W, L
    return None


def main():
    print("LoRA 5차 판정 — 원래 정책 vs 5차, 같은 에피소드 20~29")

    r1 = compare("① 망각 — 10개 태스크, 교란 없음",
                 load("outputs/v5eval_forget_base", "a_success"),
                 load("outputs/v5eval_forget_v5", "a_success"),
                 "4차는 여기서 무너졌다 (T8 90% → 10%). 5%p 넘게 떨어지면 실패로 본다")

    sw_b = load("outputs/v5eval_switch_base", "b_success")
    sw_v = load("outputs/v5eval_switch_v5", "b_success")
    fail_tags = {k for k in set(sw_b) | set(sw_v)
                 if any(k.startswith(f"A{a}_B{b}_") for a, b in
                        [(8, 0), (8, 3), (8, 5), (8, 7), (8, 9), (4, 5), (4, 9), (1, 7)])}
    ok_tags = (set(sw_b) | set(sw_v)) - fail_tags
    r2 = compare("② 원래 실패하던 전환 — 다른 조작이 필요한 쌍",
                 {k: sw_b[k] for k in fail_tags if k in sw_b},
                 {k: sw_v[k] for k in fail_tags if k in sw_v})
    r3 = compare("③ 원래 잘 되던 전환 — 나빠지면 안 된다",
                 {k: sw_b[k] for k in ok_tags if k in sw_b},
                 {k: sw_v[k] for k in ok_tags if k in sw_v})

    print("\n== 판정")
    if r1 is None:
        print("  망각 결과가 아직 없습니다.")
        return
    b1, v1, p1, _, _ = r1
    if v1 < b1 - 0.05:
        print(f"  ❌ **망각** — 10개 태스크 평균 {100 * b1:.0f}% → {100 * v1:.0f}%. 4차와 같은 실패다.")
        print("     나머지 결과는 의미가 없다 (정책 자체가 망가졌다)")
        return
    print(f"  ✅ 망각 없음 — 10개 태스크 평균 {100 * b1:.0f}% → {100 * v1:.0f}%")
    if r2:
        b2, v2, p2, w2, l2 = r2
        tag = "✅ **개선**" if (v2 > b2 and p2 < 0.05) else ("↗ 오르지만 아직 유의하지 않음" if v2 > b2 else "❌ 개선 없음")
        print(f"  {tag} — 실패하던 전환 {100 * b2:.0f}% → {100 * v2:.0f}% (짝 {w2}:{l2}, p={p2:.4f})")
    if r3:
        b3, v3, p3, w3, l3 = r3
        tag = "✅ 유지" if v3 >= b3 - 0.05 else "❌ **나빠짐**"
        print(f"  {tag} — 잘 되던 전환 {100 * b3:.0f}% → {100 * v3:.0f}% (짝 {w3}:{l3})")


if __name__ == "__main__":
    main()
