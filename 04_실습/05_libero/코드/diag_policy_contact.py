"""원래 모델이 태스크를 어떻게 해내는지 — 손끝·대상 위치·접촉·관절을 스텝마다 기록 (전문가 설계용)."""
import sys
import numpy as np
import switch_experiment as sx

task, i, target = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]   # target: 몸체 이름 (예: flat_stove_1_button)
joint = sys.argv[4] if len(sys.argv) > 4 else None
ra = sx.default_args(); ra.task_a, ra.strategy = task, "none"
r = sx.Runner(ra)
ep = sx.Episode(r, i); r.policy.reset()
m, d = ep.inner.sim.model, ep.inner.sim.data
bid = m.body_name2id(target) if target in [m.body_id2name(k) for k in range(m.nbody)] else None
tip = lambda n: d.body_xpos[m.body_name2id(n)]
from scipy.spatial.transform import Rotation as R
for s in range(300):
    act = np.asarray(ep.policy_action(r.chk_a.language), np.float32)
    ep.step(act, "A")
    T = d.body_xpos[bid] if bid is not None else ep.obj_pos(target)
    jq = float(d.qpos[m.jnt_qposadr[m.joint_name2id(joint)]]) if joint else 0.0
    con = set()
    for c in range(d.ncon):
        cc = d.contact[c]; a = m.geom_id2name(cc.geom1) or ""; b = m.geom_id2name(cc.geom2) or ""
        if ("gripper" in a) != ("gripper" in b) and "table" not in a + b:
            con.add(b if "gripper" in a else a)
    if s % 5 == 0 or con:
        e = R.from_matrix(ep.obs["robot_state"]["eef"]["mat"]).as_euler("xyz", degrees=True)
        print(f"s{s:3d} eef-T {np.round(sx.eef_pos(ep.obs) - T, 3)} euler {np.round(e, 0)} q {ep.obs['robot_state']['gripper']['qpos'][0]:.3f} "
              f"act {np.round(act, 2)} joint {jq:.3f} con {sorted(con)}", flush=True)
    if r.chk_a(ep.env):
        print("SUCCESS", s); break
