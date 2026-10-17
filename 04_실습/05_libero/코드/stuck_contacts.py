"""블록 위에서 못 내려갈 때 손(그리퍼·손목·카메라 몸통)이 무엇에 닿아 있나 (10/17).
   평가와 같은 인자로 한 장면을 다시 돌리며, 지시문 블록 위 1cm 안·블록 중심 4cm 아래에 손이 있을 때 접촉을 50스텝마다 찍는다.
   python stuck_contacts.py -- (eval_piper 인자)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import piper_sim.piper_robot as pr
pr.use_piper_in_lerobot()
import switch_experiment as sx
import piper_sim.blocks as pb

TARGET = {}
orig_step = sx.Episode.step
cnt = {"n": 0}


def step(self, action, phase):
    out = orig_step(self, action, phase)
    t = TARGET.get(phase)
    if t is None:
        return out
    obj = pb.TASKS[t][0]
    hand = sx.eef_pos(self.obs); o = self.obj_pos(obj)
    if np.linalg.norm(hand[:2] - o[:2]) < 0.01 and hand[2] - o[2] < 0.04:
        cnt["n"] += 1
        if cnt["n"] % 10 == 1:
            m, d = self.inner.sim.model, self.inner.sim.data
            pairs = set()
            for i in range(d.ncon):
                c = d.contact[i]
                a, b = m.geom_id2name(c.geom1) or f"#{c.geom1}", m.geom_id2name(c.geom2) or f"#{c.geom2}"
                if any(k in a + b for k in ("gripper", "finger", "link6", "hand", "robot0")):
                    pairs.add(tuple(sorted((a, b))))
            q = float(self.obs["robot_state"]["gripper"]["qpos"][0])
            A = getattr(self.r, "assist", None); extra = ""
            if A is not None:
                import grasp_assist as ga
                col = {v: k for k, v in ga.NAMES.items()}[obj]
                est = A.pos.get(col)
                dd = None if est is None else 100 * float(np.linalg.norm(est - hand[:2]))
                err = None if est is None else 100 * float(np.linalg.norm(est - o[:2]))
                extra = (f" | 보조: 추정-손 {dd:.2f}cm 추정-실제 {err:.2f}cm yaw_bad {getattr(A, 'yaw_bad', None)} "
                         f"dmode {getattr(A, 'dmode', None)} 동작 z {action[2]:+.2f} 그리퍼 {action[6]:+.1f}")
            print(f"[{phase} {len(self.acts)}] 손-블록 xy {100*np.linalg.norm(hand[:2]-o[:2]):.2f}cm 높이 {100*(hand[2]-o[2]):.1f}cm 그리퍼 {q:.4f} 접촉 {sorted(pairs)}{extra}", flush=True)
    return out


sx.Episode.step = step
if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    a = sx.build_parser().parse_args(argv + ["--max-steps", "600"])
    TARGET["A"] = a.task_a if a.task_b is None else a.task_a
    if a.task_b is not None:
        TARGET["B"] = a.task_b; TARGET["A2"] = a.task_a
    sys.argv = [sys.argv[0]] + argv + ["--max-steps", "600"]
    sx.main()
