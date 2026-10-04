#!/usr/bin/env python
"""성적 폴더의 28항목 평균 성공률 (10/4). python piper_sim/mean_score.py piper_b1h piper_b1ah → 줄마다 '이름 평균'"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scoreboard as sb  # noqa: E402

for base in sys.argv[1:]:
    rows = sb.collect([f"outputs/{base}_forget"], [f"outputs/{base}_switch"])
    rs = [s / n for _, _, _, s, n in rows if n]
    print(base, round(sum(rs) / len(rs), 4) if rs else 0.0)
