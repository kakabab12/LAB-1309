#!/usr/bin/env python
"""
rollback 평가 — 제약(초기 자세 복귀 금지)을 지키면서 B 성공과 A 재개가 되는가?

같이 확인하는 것
  home_dist_after_cm   전환 처리 뒤 끝단이 초기 위치에서 얼마나 떨어져 있는지
                       → retreat 은 0 에 가까워야 하고(리셋), rollback 은 커야 한다(제약 준수)
  A 재개 성공률         그 자리 전환은 0% 였다. rollback 이 이걸 깨는지가 핵심
"""
import glob
import json
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

ORDER = ["flush", "rollback", "release", "ret_rot", "ret_pos", "retreat"]


def sign_test(w, l):
    k = w + l
    return min(1.0, 2 * sum(comb(k, i) for i in range(max(w, l), k + 1)) / 2 ** k) if k else 1.0


def fmt(v, spec, dash="-"):
    return dash if v is None else format(v, spec)


def main():
    rows = defaultdict(list)
    paired = defaultdict(dict)
    for f in glob.glob("outputs/switch/A*_B*_grasp*.json"):
        d = json.load(open(f))
        a = d["args"]
        if (a.get("latency_steps", 0) or a.get("switch_n_action_steps", 0)
                or a.get("switch_noise_seed", -1) >= 0
                or a.get("dither_escape", "off") != "off"):
            continue  # 통제변수가 다른 실행은 제외
        s = a["strategy"]
        for e in d["episodes"]:
            if not e.get("switched"):
                continue
            rows[s].append(e)
            paired[(a["task_a"], a["task_b"], a["switch_at"], e["episode"])][s] = e

    print(f"{'전략':10s}{'n':>4s}{'B 성공':>8s}{'A 재개':>16s}{'둘다':>7s}"
          f"{'초기위치까지':>14s}{'되돌림':>8s}")
    out = {}
    for s in ORDER:
        r = rows.get(s)
        if not r:
            continue
        bs = [e for e in r if e.get("b_success")]
        res = [bool(e.get("a_resume_success")) for e in bs]
        hd = [e["home_dist_after_cm"] for e in r if e.get("home_dist_after_cm") is not None]
        rb = [e["to_b_rollback_steps"] for e in r if e.get("to_b_rollback_steps") is not None]
        v = {"n": len(r),
             "b": float(np.mean([bool(e.get("b_success")) for e in r])),
             "resume": float(np.mean(res)) if res else None,
             "resume_k": int(sum(res)), "resume_n": len(res),
             "both": float(np.mean([bool(e.get("both_success")) for e in r])),
             "home_cm": float(np.median(hd)) if hd else None,
             "rollback_steps": float(np.median(rb)) if rb else None}
        out[s] = v
        resume_txt = f"{100 * v['resume']:.0f}% ({v['resume_k']}/{v['resume_n']})" if res else "-"
        print(f"{s:10s}{v['n']:4d}{100 * v['b']:7.0f}%{resume_txt:>16s}{100 * v['both']:6.0f}%"
              f"{fmt(v['home_cm'], '.1f'):>12s}cm{fmt(v['rollback_steps'], '.0f'):>8s}")

    print("\n짝 비교 (같은 초기상태끼리, vs flush)")
    for s in ORDER[1:]:
        if s not in rows:
            continue
        for metric, key in [("B 성공", "b_success"), ("A 재개", "a_resume_success"), ("둘다  ", "both_success")]:
            w = l = 0
            for v in paired.values():
                if s in v and "flush" in v:
                    x, y = bool(v[s].get(key)), bool(v["flush"].get(key))
                    w += x and not y
                    l += y and not x
            print(f"  {s:10s} {metric}: {s} 만 {w:3d}, flush 만 {l:3d}, p = {sign_test(w, l):.3f}")

    Path("outputs/report").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("outputs/report/rollback.json", "w"), ensure_ascii=False, indent=2)
    print("\n저장: outputs/report/rollback.json")


if __name__ == "__main__":
    main()
