"""두 평가(28항목)를 같은 장면끼리 짝지어 비교 (10/23). 장면 번호가 같으면 배치가 같아, 한쪽만 성공한 장면 수로 부호 검정을 한다.
   python cmp_rounds.py outputs/piper_np2mc outputs/piper_np2fb [--names 예전 고침] [--loose]
   (각 경로 뒤에 _forget·_switch 를 붙여 읽음. 혼자 과제가 없으면 첫 번째 것의 혼자 과제를 같이 씀)"""
import argparse
import glob
import json
import math
import os

PAIRS = ["8:0", "8:3", "8:5", "8:7", "8:9", "4:5", "4:9", "1:7", "8:1", "8:4", "1:8", "4:1"]
RESUME = ["8:5", "8:7", "8:9", "4:5", "4:9", "1:7"]


def load(pat):
    out = {}
    for f in glob.glob(pat):
        for e in json.load(open(f)).get("episodes", []):
            out[e["episode"]] = e
    return out


def outcome(e, kind, sfx):
    if e is None:
        return None
    if kind == "단독":
        return bool(e.get("a_success" + sfx, e.get("a_success")))
    if not e.get("switched"):
        return False
    if kind == "전환":
        return bool(e.get("b_success" + sfx, e.get("b_success")))
    if kind == "재개":
        return bool(e.get("a_resume_genuine" + sfx, e.get("a_resume_genuine")))
    return bool(e.get("b_success") and e.get("a_resume_genuine") and e.get("b_still_satisfied_at_end"))     # 셋 다


def sign_p(w, l):
    """양쪽 부호 검정 p (한쪽만 성공한 장면 w 대 l)."""
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--names", nargs=2, default=["A", "B"])
    p.add_argument("--loose", action="store_true")
    a = p.parse_args()
    sfx = "_loose" if a.loose else ""
    items = [("단독", f"T{t}", f"A{t}_Bnone_*.json", "forget") for t in range(10)]
    items += [("전환", f"A{x}→B{y}", f"A{x}_B{y}_*.json", "switch") for x, y in (q.split(":") for q in PAIRS)]
    items += [("재개", f"A{x}→B{y}→A{x}", f"A{x}_B{y}_*.json", "switch") for x, y in (q.split(":") for q in RESUME)]
    items += [("셋 다", f"A{x}→B{y}→A{x}", f"A{x}_B{y}_*.json", "switch") for x, y in (q.split(":") for q in RESUME)]
    tot = {k: [0, 0, 0, 0, 0] for k in ("단독", "전환", "재개", "셋 다", "28항목")}      # A 성공, B 성공, 장면 수, B만, A만
    print(f"| 구분 | 항목 | {a.names[0]} | {a.names[1]} | {a.names[1]}만 성공 | {a.names[0]}만 성공 |")
    print("|---|---|---|---|---|---|")
    for kind, name, pat, sub in items:
        da = load(f"{a.a}_{sub}/{pat}")
        db = load(f"{a.b}_{sub}/{pat}")
        if sub == "forget" and not db:
            db = da
        eps = sorted(set(da) & set(db))
        if not eps:
            continue
        ra = [outcome(da[e], kind, sfx) for e in eps]
        rb = [outcome(db[e], kind, sfx) for e in eps]
        w = sum(y and not x for x, y in zip(ra, rb))
        l = sum(x and not y for x, y in zip(ra, rb))
        print(f"| {kind} | {name} | {sum(ra)}/{len(eps)} | {sum(rb)}/{len(eps)} | {w} | {l} |")
        for k in ([kind, "28항목"] if kind != "셋 다" else [kind]):
            t = tot[k]
            t[0] += sum(ra); t[1] += sum(rb); t[2] += len(eps); t[3] += w; t[4] += l
    print()
    for k, t in tot.items():
        if t[2]:
            print(f"{k}: {a.names[0]} {100 * t[0] / t[2]:.1f}% → {a.names[1]} {100 * t[1] / t[2]:.1f}% "
                  f"({a.names[1]}만 {t[3]}, {a.names[0]}만 {t[4]}, 부호 검정 p {sign_p(t[3], t[4]):.3f}, 장면 {t[2]})")


if __name__ == "__main__":
    main()
