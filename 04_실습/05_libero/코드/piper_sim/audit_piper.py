#!/usr/bin/env python
"""PiPER 장면에서 시범 프로그램 성공률 (teacher_audit 와 같은 방식, 2026-10-03)
   python piper_sim/audit_piper.py --tasks 0 1 2 --pairs 8:7 --episodes 2000-2009 --out outputs/piper/audit_dev.json"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import piper_sim.piper_robot as pr  # noqa: E402

pr.use_piper_in_lerobot()
import teacher_audit  # noqa: E402

if __name__ == "__main__":
    teacher_audit.main()
