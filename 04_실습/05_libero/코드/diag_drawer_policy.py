"""원래 모델이 서랍을 어떻게 여는지 — 손끝·손잡이·접촉을 기록 (T3 ep100, T0 ep102: 녹화 때 성공한 에피소드)."""
import sys
import numpy as np
import switch_experiment as sx
import task_experts as te

task, i, which = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
ra = sx.default_args(); ra.task_a, ra.strategy = task, "none"
r = sx.Runner(ra)
ep = sx.Episode(r, i); r.policy.reset()
m, d = ep.inner.sim.model, ep.inner.sim.data
tip = lambda n: d.body_xpos[m.body_name2id(n)]
prev = 0.0
for s in range(300):
    act = np.asarray(ep.policy_action(r.chk_a.language), np.float32)
    ep.step(act, "A")
    q = -te.drawer_qpos(ep, which)
    h = te.handle_pos(ep, which)
    con = set()
    for c in range(d.ncon):
        cc = d.contact[c]; a = m.geom_id2name(cc.geom1) or ""; b = m.geom_id2name(cc.geom2) or ""
        if ("gripper" in a or "robot" in a) and "cabinet" in b or ("gripper" in b or "robot" in b) and "cabinet" in a:
            con.add((a, b))
    if con or (q - prev) > 0.002 or s % 20 == 0:
        t1, t2 = tip("gripper0_finger_joint1_tip") - h, tip("gripper0_finger_joint2_tip") - h
        print(f"s{s:3d} drawer {100*q:5.1f}cm eef-h {np.round(sx.eef_pos(ep.obs)-h,3)} tip1 {np.round(t1,3)} tip2 {np.round(t2,3)} "
              f"q {ep.obs['robot_state']['gripper']['qpos'][0]:.3f} a_g {act[6]:+.1f} {sorted(con)}", flush=True)
    prev = q
    if r.chk_a(ep.env) or (which == "top" and q > 0.12 and s > 80):
        print("stop", s); break
