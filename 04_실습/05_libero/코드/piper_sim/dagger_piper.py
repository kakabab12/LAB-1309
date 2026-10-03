#!/usr/bin/env python
"""PiPER 장면에서 DAgger 교정 시범 (dagger_v6 와 같은 인자, 2026-10-03). 학습용 장면 2000번대만."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import piper_sim.piper_robot as pr  # noqa: E402

pr.use_piper_in_lerobot()
import dagger_v6  # noqa: E402

if __name__ == "__main__":
    dagger_v6.main()
