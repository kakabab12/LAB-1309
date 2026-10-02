#!/usr/bin/env python
"""PiPER 장면에서 시범 수집 (collect_v6 와 같은 방식, 2026-10-03). 학습용 장면은 2000번대만 쓴다.
   python piper_sim/collect_piper.py normal --tasks 0 1 --episodes 2000-2059 --dart 0.1 --out data/piper_normal
   python piper_sim/collect_piper.py switch --pairs 8:0,8:3 --episodes 2000-2029 --dart 0.1 --out data/piper_switch"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import piper_sim.piper_robot as pr  # noqa: E402

pr.use_piper_in_lerobot()
import collect_v6  # noqa: E402

if __name__ == "__main__":
    collect_v6.main()
