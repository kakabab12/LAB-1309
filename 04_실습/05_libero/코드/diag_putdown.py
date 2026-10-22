"""내려놓기 보조(⑫)가 언제 켜지나 (10/22). 돌아오기 '판 밖에 놓음' 중 원래 블록이 목표에서 30~55cm 떨어진 장면이 있고,
   그 장면들은 내려놓기가 3~4번 켜졌다 → 보조가 쥔 블록을 잘못 알아(지시문의 블록인데 다른 블록으로) 원래 블록을 빈 자리에
   내려놓는다는 가설. 켜질 때마다 단계·지시문·보조가 본 쥔 블록·실제로 손가락에 닿은 블록·블록 위치(실제/보조 기억)를 찍는다.
   python diag_putdown.py -- (eval_piper 인자)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import piper_sim.piper_robot as pr
pr.use_piper_in_lerobot()
import switch_experiment as sx
import grasp_assist as ga
import dagger_v6 as dv

COL = {v: k for k, v in ga.NAMES.items()}
cur = {"phase": None, "ep": None}
orig_step = sx.Episode.step


def step(self, action, phase):
    cur["phase"], cur["ep"] = phase, self
    return orig_step(self, action, phase)


orig_call = ga.GraspAssist.__call__


def call(self, ep, act, instruction):
    was = getattr(self, "pd", None)
    held_before = getattr(self, "held", None)
    out = orig_call(self, ep, act, instruction)
    if was is None and getattr(self, "pd", None) is not None:
        real = dv.held_object(ep)
        pos = {COL[o]: np.round(ep.obj_pos(o)[:2], 3) for o in ga.NAMES.values() if COL[o] in ga.objdet.BLOCKS}
        mem = {c: np.round(v, 3) for c, v in self.pos.items() if c in ga.objdet.BLOCKS}
        hand = np.round(sx.eef_pos(ep.obs), 3)
        print(f"[{cur['phase']} {len(ep.acts)}] 내려놓기 켜짐: 지시 '{instruction}' 보조가 본 쥔 블록 {self.held} (직전 {held_before}) "
              f"실제 손가락에 닿은 것 {COL.get(real, real)} | 손 {hand} 자리 {np.round(self.pd['spot'], 3)}\n      실제 {pos}\n      기억 {mem}", flush=True)
    return out


sx.Episode.step = step
ga.GraspAssist.__call__ = call
if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    sys.argv = [sys.argv[0]] + argv + ["--max-steps", "600"]
    sx.main()
