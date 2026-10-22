"""보조가 쥔 블록을 무엇으로 알아보나 (10/23 새벽). 고친 보조에서 노란 막대 B 가 다시 실패한 장면(A8→B7 1006)은 막대를 가운데로
   쥐었는데 내려놓기(⑫)가 한 번 더 켜졌다 → 쥔 블록을 노랑이 아닌 것으로 알아봤다는 가설. 보조의 '쥔 블록'이 바뀔 때마다·내려놓기가
   켜질 때마다: 단계·지시·보조가 본 쥔 블록·실제 손가락에 닿은 것·손·사진으로 잰 블록 위치(손까지 거리)·기억을 찍는다.
   python diag_held.py -- (eval_piper 인자)"""
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
cur = {"phase": None}
orig_step = sx.Episode.step


def step(self, action, phase):
    cur["phase"] = phase
    return orig_step(self, action, phase)


orig_call = ga.GraspAssist.__call__


def call(self, ep, act, instruction):
    held0, pd0 = getattr(self, "held", None), getattr(self, "pd", None)
    hr0 = getattr(self, "n_hold_release", 0)
    out = orig_call(self, ep, act, instruction)
    if getattr(self, "n_hold_release", 0) > hr0 and (hr0 - getattr(self, "hr_last", -99) > 1 or len(ep.acts) % 20 == 0):
        dst = self._dest(instruction)
        hand = np.asarray(ep.obs["robot_state"]["eef"]["pos"], dtype=np.float64)
        dm = self.pos.get(dst)
        print(f"[{cur['phase']} {len(ep.acts)}] ⑤ 벌리기 막음: 손 {np.round(hand, 3).tolist()} 놓을 곳({dst}) 기억 {None if dm is None else np.round(dm, 3).tolist()} "
              f"거리 {None if dm is None else round(100 * float(np.linalg.norm(dm - hand[:2])), 1)}cm 쥔 {self.held}", flush=True)
    if getattr(self, "n_hold_release", 0) > hr0:
        self.hr_last = hr0
    held1, pd1 = getattr(self, "held", None), getattr(self, "pd", None)
    if held1 != held0 or (pd0 is None and pd1 is not None):
        real = dv.held_object(ep)
        hand = np.asarray(ep.obs["robot_state"]["eef"]["pos"], dtype=np.float64)
        q = float(ep.obs["robot_state"]["gripper"]["qpos"][0])
        fr = {c: f"{np.round(v, 3).tolist()}({100 * np.linalg.norm(v - hand[:2]):.1f})" for c, v in self.fresh.items() if c in ga.objdet.BLOCKS}
        mem = {c: np.round(v, 3).tolist() for c, v in self.pos.items() if c in ga.objdet.BLOCKS}
        true = {c: np.round(ep.obj_pos(n)[:2], 3).tolist() for c, n in ga.NAMES.items()}
        what = "내려놓기 켜짐" if (pd0 is None and pd1 is not None) else f"쥔 블록 {held0} → {held1}"
        print(f"[{cur['phase']} {len(ep.acts)}] {what} | 지시 '{instruction}' 실제 손가락 {COL.get(real, real)} 그리퍼 {q:.3f} 손 {np.round(hand, 3).tolist()}"
              f"\n      사진(손까지cm) {fr}\n      기억 {mem}\n      실제 {true}" + (f"\n      내려놓을 자리 {np.round(pd1['spot'], 3).tolist()} z_back {pd1.get('z_back')}" if pd1 is not None and pd0 is None else ""), flush=True)
    return out


sx.Episode.step = step
ga.GraspAssist.__call__ = call
if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    sys.argv = [sys.argv[0]] + argv + ["--max-steps", "600"]
    sx.main()
