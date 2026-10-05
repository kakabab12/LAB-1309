"""시범을 눈 감고 다시 실행: 변화량 그대로 vs 절대 목표 위치(지금 손과의 차이로 명령). 원래 궤적과의 차이·성공."""
import sys, glob, numpy as np
sys.path.insert(0, "/home/user/smolVLA")
import piper_sim.piper_robot as pr
pr.use_piper_in_lerobot()
import sim_only as so, switch_experiment as sx
res = {"delta": [], "abs": []}
for f in sorted(glob.glob("/home/user/smolVLA/data/piper_blk_normal/episodes/N8_ep20[0-1][0-9].npz"))[:12]:
    d = np.load(f, allow_pickle=True); i = int(f.split("ep")[-1][:4])
    A, P = d["action"].astype(np.float32), d["eef_pos"]
    for mode in ("delta", "abs"):
        r = so.SimRunner(8); ep = sx.Episode(r, i); ep.inner.horizon = 4000
        err = []
        for t in range(len(A)):
            a = A[t].copy()
            if mode == "abs":
                tgt = P[t] + 0.05 * np.clip(a[:3], -1, 1)
                a[:3] = np.clip((tgt - sx.eef_pos(ep.obs)) / 0.05, -1, 1)
            err.append(np.linalg.norm(sx.eef_pos(ep.obs) - P[t]))
            ep.step(a, "R")
        for _ in range(15): ep.step(np.r_[np.zeros(6), -1.0], "R")
        res[mode].append((r.chk_a(ep.env), max(err), np.mean(err)))
        ep.env.close()
    print(i, "delta", res["delta"][-1][0], round(100 * res["delta"][-1][1], 1), "| abs", res["abs"][-1][0], round(100 * res["abs"][-1][1], 1), flush=True)
for m in res:
    v = res[m]; print(m, "성공", sum(x[0] for x in v), "/", len(v), "최대 오차 평균 cm", round(100 * np.mean([x[1] for x in v]), 2))
