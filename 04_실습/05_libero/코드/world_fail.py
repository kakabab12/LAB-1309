#!/usr/bin/env python
"""여러 모양·N개 장면 평가(world_eval json) 실패 나누기 (10/25). 일마다 world_eval.Track 기록으로:
   성공(3cm) / 판 안이지만 3cm 밖 / 쥐었지만 못 놓음(놓을 곳에서 먼 곳) / 다가갔지만 못 쥠(손 5cm 안) / 다가가지도 못함.
   python piper_sim/world_fail.py outputs/world_w1b [outputs/world_w1ba ...]"""
import json
import sys
from collections import Counter
from pathlib import Path

KINDS = ["성공 (3cm)", "판 안, 3cm 밖", "쥐었지만 못 놓음", "다가갔지만 못 쥠", "다가가지도 못함", "기록 없음"]


def kind(t):
    if t.get("ok"):
        return KINDS[0]
    if t.get("loose"):
        return KINDS[1]
    if "held_steps" not in t:
        return KINDS[5]
    if t["held_steps"] > 0:
        return KINDS[2]
    if t["hand_min_cm"] <= 5.0:
        return KINDS[3]
    return KINDS[4]


def main(dirs):
    for d in dirs:
        c, n = Counter(), 0
        for f in sorted(Path(d).glob("*.json")):
            for e in json.loads(f.read_text()):
                for t in e["tasks"]:
                    c[kind(t)] += 1
                    n += 1
        print(f"\n{d} (일 {n}개)")
        for k in KINDS:
            if c[k]:
                print(f"  {k:18s} {c[k]:4d}  ({100 * c[k] / n:.0f}%)")


if __name__ == "__main__":
    main(sys.argv[1:])
