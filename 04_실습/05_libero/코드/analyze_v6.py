#!/usr/bin/env python
"""
6차 판정 — 조건마다 성공률, 95% 신뢰구간(Wilson), 85% 목표 달성 여부. 원래 모델과 같은 장면에서 나란히.

  ① 10개 태스크 단독      outputs/v6eval_forget        (원래: v5eval_forget_base 20~29 + v6eval_forget_base 30~49)
  ② 전환 12쌍 B 성공       outputs/v6eval_switch
  ③ 원래 일로 돌아가기     ② 와 같은 실행에서, B 를 마친 뒤 A 를 진짜로 다시 해냈는가 (0스텝 우연 제외)
"""
import glob
import json
import math
from pathlib import Path


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def load(dirs, key):
    out = {}
    for d in dirs:
        for f in glob.glob(f"{d}/*.json"):
            js = json.load(open(f))
            tag = Path(f).stem
            for e in js.get("episodes", []):
                if key == "b_success" and not e.get("switched"):
                    continue
                v = bool(e.get(key))
                out.setdefault(tag, {})[e["episode"]] = v
    return out


def table(title, new, base, target=0.85):
    print(f"\n== {title}")
    print(f"{'조건':<28}{'6차':>14}{'95% 구간':>16}{'85%':>6}{'원래':>12}")
    K = N = 0
    for tag in sorted(new):
        v = new[tag]
        k, n = sum(v.values()), len(v)
        lo, hi = wilson(k, n)
        b = base.get(tag, {})
        common = [e for e in b if e in v]
        bs = f"{100 * sum(b[e] for e in common) / len(common):.0f}% ({len(common)})" if common else "-"
        print(f"{tag:<28}{k:>5}/{n:<3}{100 * k / max(n, 1):5.0f}%  [{100 * lo:3.0f}–{100 * hi:3.0f}%]"
              f"{'  O' if k / max(n, 1) >= target else '  X':>6}{bs:>12}")
        K += k
        N += n
    if N:
        lo, hi = wilson(K, N)
        print(f"{'합계':<28}{K:>5}/{N:<3}{100 * K / N:5.0f}%  [{100 * lo:3.0f}–{100 * hi:3.0f}%]")


# 원래 일로 돌아가기가 의미 있는 조합만 (나머지는 B 가 A 를 되돌리거나 막는다)
#   8:1 8:4 1:8 4:1  같은 그릇을 다른 곳에 → A 로 돌아가면 B 가 풀린다
#   8:3              그릇을 서랍에 넣는다 → A 의 그릇을 B 가 써 버린다
#   8:0              열린 가운데 서랍이 접시를 덮는다 → 서랍을 닫지 않고는 접시에 놓을 수 없다 (전문가도 0/10)
RESUME_OK = ("A8_B5_", "A8_B7_", "A8_B9_", "A4_B5_", "A4_B9_", "A1_B7_")


def naturalness(dirs):
    """B 에 성공한 에피소드의 전환 뒤 구간: 멈춤 횟수, 우회 비율, jerk, 낙하, 초기 자세 최소 거리."""
    import numpy as np
    from naturalness import metrics
    rows = {}
    for d in dirs:
        for f in glob.glob(f"{d}/*.json"):
            js = json.load(open(f))
            tag = Path(f).stem
            for e in js.get("episodes", []):
                if not e.get("switched") or not e.get("b_success"):
                    continue
                tf = Path(d) / "traj" / f"{tag}_ep{e['episode']}.npz"
                if not tf.exists():
                    continue
                z = np.load(tf, allow_pickle=True)
                ph = [str(x) for x in z["phase"]]
                if "B" not in ph:
                    continue
                ts = ph.index("B")
                m = metrics(z["pos"], ph, ts)
                if m is None:
                    continue
                mh = float(np.linalg.norm(z["pos"][ts:] - z["home_pos"], axis=1).min()) * 100
                r = rows.setdefault(tag, {"stops": [], "detour": [], "jerk": [], "drop": [], "home": []})
                r["stops"].append(m["stops"]); r["detour"].append(m["detour"])
                r["jerk"].append(e.get("jerk") or np.nan); r["drop"].append(bool(e.get("drop"))); r["home"].append(mh)
    return rows


def nat_table(new, base):
    import numpy as np
    print("\n== ④ 자연스러움 (B 성공 에피소드, 전환 뒤 구간) — 중앙값, 초기 자세는 최솟값")
    print(f"{'조건':<24}{'멈춤':>10}{'우회 비율':>14}{'jerk':>14}{'낙하':>10}{'초기 자세 최소':>16}")
    for tag in sorted(set(new) | set(base)):
        def f(r, k, fn=np.nanmedian):
            return fn(r[k]) if r and r[k] else float("nan")
        n, b = new.get(tag), base.get(tag)
        print(f"{tag:<24}{f(n,'stops'):>5.0f}/{f(b,'stops'):<4.0f}{f(n,'detour'):>7.2f}/{f(b,'detour'):<6.2f}"
              f"{f(n,'jerk'):>7.1f}/{f(b,'jerk'):<6.1f}{100*f(n,'drop',np.mean):>5.0f}/{100*f(b,'drop',np.mean):<4.0f}"
              f"{f(n,'home',np.min):>8.1f}/{f(b,'home',np.min):<6.1f}cm")
    print("   (6차/원래. 초기 자세 최소가 7cm 미만이면 제약 위반)")


def main():
    table("① 10개 태스크 단독", load(["outputs/v6eval_forget"], "a_success"),
          load(["outputs/v5eval_forget_base", "outputs/v6eval_forget_base"], "a_success"))
    table("② 전환 (B 성공)", load(["outputs/v6eval_switch"], "b_success"),
          load(["outputs/v5eval_switch_base", "outputs/v6eval_switch_base"], "b_success"))
    keep = lambda d: {k: v for k, v in d.items() if k.startswith(RESUME_OK)}
    table("③ 원래 일로 돌아가기 (진짜 재개, 충돌 없는 6개 조합)",
          keep(load(["outputs/v6eval_switch"], "a_resume_genuine")),
          keep(load(["outputs/v5eval_switch_base", "outputs/v6eval_switch_base"], "a_resume_genuine")))
    nat_table(naturalness(["outputs/v6eval_switch"]),
              naturalness(["outputs/v5eval_switch_base", "outputs/v6eval_switch_base"]))


if __name__ == "__main__":
    main()
