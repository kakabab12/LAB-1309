"""돌아오기(A2) 실패 중 '판이 밀림'을 본다 (10/22 밤). 새 기준 실패 9개 중 4개는 블록을 판의 원래 자리 1~2cm 안에 놓았는데
   판이 4~6cm 밀려나 있었다. 보조는 판 위치를 처음 한 번만 재므로(grasp_assist._update) 밀린 판의 옛 자리에 놓는다는 가설.
   B·A2 동안 A 의 판이 3mm 넘게 움직일 때마다: 단계·손·그리퍼·판에 닿은 것·A 블록 위치·보조 기억(판·블록)을 찍는다.
   A 블록을 놓을 때(그리퍼가 열릴 때)도 실제 판·기억 판까지 거리를 찍는다.
   python diag_pad.py -- (eval_piper 인자)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import piper_sim.piper_robot as pr
pr.use_piper_in_lerobot()
import switch_experiment as sx
import grasp_assist as ga
import piper_sim.blocks as pb

COL = {v: k for k, v in ga.NAMES.items()}
COL.update(plate_1="purple", gray_pad_1="gray", orange_pad_1="orange", white_pad_1="white")
st = {"pad": None, "q": None, "ep": None}
orig_step = sx.Episode.step


def contacts(ep, name):
    m, d = ep.inner.sim.model, ep.inner.sim.data
    out = set()
    for i in range(d.ncon):
        c = d.contact[i]
        a, b = m.geom_id2name(c.geom1) or "", m.geom_id2name(c.geom2) or ""
        if a.startswith(name) and not b.startswith(name):
            out.add(b)
        elif b.startswith(name) and not a.startswith(name):
            out.add(a)
    return sorted(out)


def step(self, action, phase):
    out = orig_step(self, action, phase)
    if st["ep"] is not self:
        st.update(pad=None, q=None, ep=self)
    if phase not in ("B", "A2"):
        return out
    obj, tgt, _ = pb.TASKS[self.a.task_a]
    p, o = self.obj_pos(tgt), self.obj_pos(obj)
    hand = sx.eef_pos(self.obs)
    q = float(self.obs["robot_state"]["gripper"]["qpos"][0])
    A = getattr(self.r, "assist", None)
    mem = getattr(A, "pos", {}) if A is not None else {}
    fr = getattr(A, "fresh", {}) if A is not None else {}
    mp, mo = mem.get(COL[tgt]), mem.get(COL[obj])
    fmt = lambda v: None if v is None else np.round(v[:2], 3).tolist()
    if st["pad"] is None:
        st["pad"] = p.copy()
    if np.linalg.norm(p[:2] - st["pad"][:2]) > 0.003:
        print(f"[{phase} {len(self.acts)}] 판 움직임 {100 * np.linalg.norm(p[:2] - st['pad'][:2]):.1f}cm → {fmt(p)} | 손 {np.round(hand, 3).tolist()} 그리퍼 {q:.3f} "
              f"판에 닿은 것 {contacts(self, tgt)} | 블록 {np.round(o, 3).tolist()} | 기억 판 {fmt(mp)} 블록 {fmt(mo)} 사진 판 {fmt(fr.get(COL[tgt]))} "
              f"보조 쥔 {getattr(A, 'held', None)} 내려놓는 중 {getattr(A, 'pd', None) is not None}", flush=True)
        st["pad"] = p.copy()
    if phase == "A2" and st["q"] is not None and st["q"] < 0.028 <= q:      # 그리퍼가 열림 = 놓음
        print(f"[A2 {len(self.acts)}] 놓음: 블록 {np.round(o, 3).tolist()} 실제 판까지 {100 * np.linalg.norm(o[:2] - p[:2]):.1f}cm "
              f"기억 판까지 {None if mp is None else round(100 * float(np.linalg.norm(o[:2] - mp)), 1)}cm | 기억 블록 {fmt(mo)} 사진 블록 {fmt(fr.get(COL[obj]))} "
              f"손 {np.round(hand, 3).tolist()}", flush=True)
    if len(self.acts) % 20 == 0 and A is not None:     # B·A2 20스텝마다 보조 상태 (맴돎·헛손질 원인 찾기). 그 단계 지시문의 블록 기준
        bo = obj if phase == "A2" else pb.TASKS[self.a.task_b][0]
        c = COL[bo]
        o, mo = self.obj_pos(bo), mem.get(c)
        try:
            m = A._wrist_angle(self.obs["pixels"]["image2"], c)
        except Exception:
            m = None
        j6 = float(np.degrees(self.obs["robot_state"]["joints"]["pos"][5]))
        print(f"   ({phase} {len(self.acts)}) 손 {np.round(hand, 3).tolist()} 그리퍼 {q:.3f} | 블록 실제 {fmt(o)} 기억 {fmt(mo)} 사진 {fmt(fr.get(c))} "
              f"기억-실제 {None if mo is None else round(100 * float(np.linalg.norm(mo - o[:2])), 1)}cm | 손목 어긋남 {None if m is None else round(m, 1)}° "
              f"6번 관절 {j6:.0f}° yaw_bad {getattr(A, 'yaw_bad', None)} 내려가기 {getattr(A, 'dmode', None)} 막힘풀기 {getattr(A, 'ujmode', None)} "
              f"쥔 {getattr(A, 'held', None)}", flush=True)
    st["q"] = q
    return out


sx.Episode.step = step
if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    sys.argv = [sys.argv[0]] + argv + ["--max-steps", "600"]
    sx.main()
