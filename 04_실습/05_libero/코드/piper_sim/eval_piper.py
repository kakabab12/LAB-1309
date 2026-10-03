#!/usr/bin/env python
"""PiPER 장면에서 학생 모델 평가 (switch_experiment 와 같은 인자, 2026-10-03)
   python piper_sim/eval_piper.py --policy outputs/piper_s1_model/merged --task-a 8 --strategy none \
       --episodes 10 --start-episode 1000 --latency-steps 11 --ttrtc --n-action-steps 1 --out outputs/piper_s1h"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import piper_sim.piper_robot as pr  # noqa: E402

pr.use_piper_in_lerobot()
import switch_experiment  # noqa: E402

# PiPER 평가 시간 제한: 단계(A, B, 재개)마다 600스텝(30초). 기본 300스텝이면 PiPER 시범 프로그램도
#   T3(위 서랍 + 그릇, 보통 448·최대 558스텝)·A8→B3(488스텝)을 못 끝낸다 (10/3 시범 길이 측정)
MAX_STEPS = "600"

if __name__ == "__main__":
    if "--max-steps" not in sys.argv:
        sys.argv += ["--max-steps", MAX_STEPS]
    switch_experiment.main()
