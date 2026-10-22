"""쌓아 둔 B(파랑 위 초록)가 돌아오기(A2) 중 언제·왜 무너지나 (10/22). --wrong-guard 로도 그대로라 원인을 다시 본다.
   A2 에서 초록이 파랑 위(옆 2cm·높이차 3.5cm 넘게)에서 벗어나는 순간: 손 위치·그리퍼·초록에 닿은 것·보조 상태를 찍는다.
   python diag_stack.py -- (eval_piper 인자, 예: --task-a 4 --task-b 9 ...)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import piper_sim.piper_robot as pr
pr.use_piper_in_lerobot()
import switch_experiment as sx

G, B = "cream_cheese_1", "blue_block_1"
st = {"on": None, "n": 0}
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
    if phase not in ("B", "A2"):
        return out
    g, b = self.obj_pos(G), self.obj_pos(B)
    on = np.linalg.norm(g[:2] - b[:2]) < 0.02 and g[2] - b[2] > 0.035
    hand = sx.eef_pos(self.obs)
    A = getattr(self.r, "assist", None)
    if st["on"] is not None and on != st["on"]:
        print(f"[{phase} {len(self.acts)}] 초록 {'쌓임' if on else '무너짐'}: 손 {np.round(hand, 3)} 그리퍼 {float(self.obs['robot_state']['gripper']['qpos'][0]):.4f} "
              f"동작 {np.round(action, 2)} 초록에 닿은 것 {contacts(self, G)} | 보조 쥔 {getattr(A, 'held', None)} 내려놓는 중 {getattr(A, 'pd', None) is not None} "
              f"초록 {np.round(g, 3)} 파랑 {np.round(b, 3)}", flush=True)
    if not on and st["on"] and phase == "A2":
        st["n"] = 25          # 무너진 뒤 25스텝 동안 5스텝마다 더 찍음
    if st["n"] > 0 and st["n"] % 5 == 0:
        print(f"   [{phase} {len(self.acts)}] 손 {np.round(hand, 3)} 초록 {np.round(g, 3)} 닿음 {contacts(self, G)}", flush=True)
    st["n"] = max(0, st["n"] - 1)
    st["on"] = on
    return out


sx.Episode.step = step
if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    sys.argv = [sys.argv[0]] + argv + ["--max-steps", "600"]
    sx.main()
