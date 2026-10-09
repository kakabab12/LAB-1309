"""Why is the pallet expert 6 mm off at slot 5? Per episode: IK error, bin offset before release, final dxy.
usage: [PALLET_IK_POS=1] PALLET_X0=0.20 PALLET_Y0=0.017 PALLET_GAP_Y=0.014 PALLET_VIA=far ACT_DESIGN=v4 python pallet_slot5_diag.py 4"""
import math, sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import gen_pallet as GP, pallet_grid as PG, stack_env as SE
slots = PG.configure(**GP.LAYOUT)
env = PG.make_env(slots, render=False)
k = int(sys.argv[1]) if len(sys.argv) > 1 else 4
rng = np.random.default_rng(21000 + 1000 * k)
ex = PG.GridExpert(env, rng); home = ex.home_q()
s = slots[k]
print("slot", k, "target xy", s["xy"], "layer", s["layer"], "row", s["row"], "col", s["col"])
for ep in range(8):
    env.reset(GP.scene(slots, k, rng), home)
    refs = {t["name"]: env.bin_state(t["name"]).pos.copy() for t in slots[:k]}
    bs = env.bin_state(s["name"]); tgt = PG.slot_target(env, slots, k)
    traj = ex.plan(env.qpos(), bs.pos, bs.yaw, tgt, upper=s["layer"] > 0, far=s["row"] == 1)
    # the release starts where the gripper command first opens after the carry; record the bin pos just before it
    g = traj[:, 5]; closed = g < (SE.GRIPPER_OPEN + SE.GRIPPER_CLOSED) / 2 if SE.GRIPPER_CLOSED < SE.GRIPPER_OPEN else g > (SE.GRIPPER_OPEN + SE.GRIPPER_CLOSED) / 2
    idx = np.where(closed)[0]; rel = idx[-1] if len(idx) else len(traj) - 1
    pre = None; fk_err = None
    for t, act in enumerate(traj):
        env.step(act)
        if t == rel:
            pre = env.bin_state(s["name"]).pos.copy()
            p_cmd, _ = ex.ik.fk(act); p_act, _ = ex.ik.fk(env.qpos())
            fk_err = 1000 * (p_act - p_cmd)
    env.settle(10)
    r = PG.evaluate(env, slots, k, refs, tgt)
    fin = env.bin_state(s["name"]).pos
    print(ep, "ok" if r["success"] else "FAIL", "ik_err %.1fmm" % (1000 * ex.last_ik_err),
          "pre-release dxy", (1000 * (pre[:2] - tgt[:2])).round(1), "final dxy", (1000 * (fin[:2] - tgt[:2])).round(1),
          "tracking err (act-cmd) mm", fk_err.round(1))
