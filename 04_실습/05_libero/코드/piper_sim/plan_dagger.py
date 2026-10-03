#!/usr/bin/env python
"""PiPER 라운드: 95% 미만 항목을 성적에 따라 나눠 교정 시범(DAgger) 장면 수를 정한다 (10/3).
   50% 미만 → 30장면, 50~85% → 20장면, 85~95% → 10장면. bash 에서 eval 할 변수를 찍는다.
   python piper_sim/plan_dagger.py --base piper_s1h --start 2100"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scoreboard as sb  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--start", type=int, default=2100)
    p.add_argument("--target", type=float, default=0.95)
    a = p.parse_args()
    rows = sb.collect([f"outputs/{a.base}_forget"], [f"outputs/{a.base}_switch"])
    groups = {"T": {30: [], 20: [], 10: []}, "P": {30: [], 20: [], 10: []}, "R": {30: [], 20: [], 10: []}}
    for g, key, _, s, n in rows:
        if n == 0 or s / n >= a.target:
            continue
        r = s / n
        k = 30 if r < 0.5 else (20 if r < 0.85 else 10)
        if g == "단독":
            groups["T"][k].append(key[1:])
        else:
            x = key.replace("A", "").replace("B", "").split("→")
            groups["P" if g == "전환" else "R"][k].append(f"{x[0]}:{x[1]}")
    for kind, gs in groups.items():
        for k, items in gs.items():
            sep = " " if kind == "T" else ","
            print(f'{kind}{k}="{sep.join(items)}"')
            print(f'E{k}="{a.start}-{a.start + k - 1}"')


if __name__ == "__main__":
    main()
