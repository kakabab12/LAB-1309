"""10개 태스크 전문가 성적표 (모델 없이). 쓰는 법: python test_all_experts.py <태스크> <에피소드 수> [시작]"""
import json, sys
import numpy as np
import sim_only as so, switch_experiment as sx, task_experts as te
task, n = int(sys.argv[1]), int(sys.argv[2]); s0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
import scripted_expert as _se; _se.DART_SIGMA = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
r = so.SimRunner(task); res = []
for i in range(s0, s0 + n):
    ep = sx.Episode(r, i); n0 = len(ep.log["pos"])
    te.EXPERT[task](ep)
    ok = False
    for _ in range(15):
        if r.chk_a(ep.env): ok = True; break
        ep.step(np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32), "E")
    res.append({"ep": i, "ok": ok, "steps": len(ep.log["pos"]) - n0})
    ep.env.close()
k = sum(x["ok"] for x in res)
json.dump(res, open(f"outputs/expert/all_T{task}_s{_se.DART_SIGMA}.json", "w"))
print(f"== T{task} 잡음 {_se.DART_SIGMA}: {k}/{n}  스텝 중앙값 {int(np.median([x['steps'] for x in res]))}", flush=True)
