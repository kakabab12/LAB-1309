#!/usr/bin/env python
"""
증폭이 정말 "귀를 열어주는가" — 여러 조건의 CMI 를 나란히 비교

왜 이걸 보나
  전환 성공률이 안 오르더라도, CMI 가 올랐는지로 원인이 갈린다.

    CMI ↑ 인데 성공 그대로  →  채널은 열렸는데 행동이 안 따라온다
                              = **지시문 채널이 병목이 아니다.** 가설을 고쳐야 한다
    CMI 도 그대로            →  우리 증폭이 채널을 실제로 못 키운다
                              = 증폭 방식을 바꿔야 한다 (토큰 반복, attention 조정 …)

  어느 쪽이든 다음에 할 일이 정해진다.

⚠️ 공정한 비교의 조건
  증폭을 켜면 **A 의 지시문만 음의 조건과 같아져 증폭을 안 받는다.**
  그래서 증폭 실험은 A 지시문을 뺀 9개로 잰다.
  **기준선도 반드시 9개(--exclude-own)로 재야** 비교가 된다 — M 개수가 다르면 CMI 가 다르게 나온다.
  이 스크립트는 그 조건이 어긋나면 경고한다.

쓰는 법
  python compare_steer.py outputs/steer_base9 outputs/steer_cfg15 outputs/steer_cfg30
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

PROBES = ["start", "grasp3", "grasp20"]
LABEL = {"start": "에피소드 시작", "grasp3": "잡은 직후", "grasp20": "들고 이동 중"}


def load(d):
    f = Path(d) / "steer.json"
    if not f.exists():
        return None
    j = json.load(open(f))
    a = j.get("args", {})
    return {"rows": j["rows"], "w": a.get("cfg_w", 1.0),
            "excl": bool(a.get("exclude_own", False)),
            "m": a.get("n_instructions", 10) - int(bool(a.get("exclude_own", False))),
            "name": Path(d).name}


def main():
    dirs = sys.argv[1:] or ["outputs/steer_base9", "outputs/steer_cfg15", "outputs/steer_cfg30"]
    ds = [x for x in (load(d) for d in dirs) if x]
    if not ds:
        print("결과가 없습니다:", dirs)
        return

    ms = {d["m"] for d in ds}
    if len(ms) > 1:
        print(f"⚠️ **비교 불가**: 지시문 개수가 다릅니다 {[(d['name'], d['m']) for d in ds]}")
        print("   --exclude-own 을 붙여 같은 개수로 다시 재야 합니다.\n")

    print(f"== CMI (nats) — 지시문 {sorted(ms)} 개\n")
    hdr = f"{'측정 지점':<14}" + "".join(f"{d['name'][:14]:>16}" for d in ds)
    print(hdr)
    print(f"{'':14}" + "".join(f"{'w=' + f'{d[chr(119)]:g}':>16}" for d in ds))
    vals = {}
    for p in PROBES:
        line = f"{LABEL[p]:<14}"
        any_row = False
        for d in ds:
            s = [r["cmi_nats"] for r in d["rows"] if r["probe"] == p]
            vals[(d["name"], p)] = s
            line += f"{np.mean(s):16.4f}" if s else f"{'-':>16}"
            any_row = any_row or bool(s)
        if any_row:
            print(line)

    base = ds[0]
    print(f"\n== 기준선({base['name']}) 대비 변화")
    for d in ds[1:]:
        print(f"\n  {d['name']} (w={d['w']:g})")
        for p in PROBES:
            a, b = vals[(base["name"], p)], vals[(d["name"], p)]
            if not a or not b:
                continue
            diff = np.mean(b) - np.mean(a)
            pct = 100 * diff / max(abs(np.mean(a)), 1e-9)
            # 같은 (태스크, 에피소드) 끼리 짝지을 수 있으면 짝 검정
            ka = {(r["task_a"], r["episode"]): r["cmi_nats"]
                  for r in base["rows"] if r["probe"] == p}
            kb = {(r["task_a"], r["episode"]): r["cmi_nats"]
                  for r in d["rows"] if r["probe"] == p}
            common = sorted(set(ka) & set(kb))
            ptxt = ""
            if len(common) >= 6:
                x = [kb[k] - ka[k] for k in common]
                if any(v != 0 for v in x):
                    ptxt = f"  (짝 검정 n={len(common)}, p={wilcoxon(x).pvalue:.3f})"
            print(f"    {LABEL[p]:<14}{np.mean(a):8.4f} → {np.mean(b):8.4f}"
                  f"  ({diff:+.4f}, {pct:+.0f}%){ptxt}")

    print("\n== 판정 (핵심은 '잡은 직후')")
    g = "grasp3"
    a = vals[(base["name"], g)]
    if not a:
        print("'잡은 직후' 측정이 없습니다.")
        return
    best = max(ds[1:], key=lambda d: np.mean(vals[(d["name"], g)] or [-9]), default=None)
    if best is None or not vals[(best["name"], g)]:
        print("증폭 조건 결과가 없습니다.")
        return
    b = vals[(best["name"], g)]
    rel = (np.mean(b) - np.mean(a)) / max(abs(np.mean(a)), 1e-9)
    print(f"가장 좋은 조건 {best['name']} (w={best['w']:g}): "
          f"{np.mean(a):.4f} → {np.mean(b):.4f} ({100 * rel:+.0f}%)")
    start = np.mean(vals[(base["name"], "start")] or [np.nan])
    if rel > 0.2:
        print("→ ✅ 증폭이 **실제로 지시문 채널을 키운다.**")
        if not np.isnan(start):
            print(f"   빈손 상태({start:.3f})의 {100 * np.mean(b) / start:.0f}% 까지 회복")
        print("   전환 성공률이 안 올랐다면, **채널이 병목이 아니라는 뜻**이다 — 가설을 고쳐야 한다")
    elif rel > -0.1:
        print("→ ❌ 증폭이 CMI 를 **거의 못 올린다.**")
        print("   속도장을 외삽해도 지시문의 '영향력' 자체는 안 커진다는 뜻이다")
        print("   → 다른 증폭 방법을 봐야 한다: 지시문 토큰 반복, attention 가중치 조정,")
        print("      또는 CMI 를 목적함수로 하는 학습")
    else:
        print("→ ⚠️ 증폭이 CMI 를 오히려 **낮춘다.** 외삽이 동작을 망가뜨리고 있을 수 있다")
        print("   (동작 크기가 포화 구간에 들어가면 지시문 차이가 묻힌다)")


if __name__ == "__main__":
    main()
