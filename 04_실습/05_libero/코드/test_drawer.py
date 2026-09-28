"""서랍 전문가 시험 — 모델 없이 (GPU 거의 안 씀)."""
import sys
import numpy as np
import sim_only as so
import switch_experiment as sx
import task_experts as te

which = sys.argv[1] if len(sys.argv) > 1 else "top"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
task = 0 if which == "middle" else 3
r = so.SimRunner(0)
chk0 = sx.GoalChecker(r.suite, 0)
res = []
for i in range(n):
    ep = sx.Episode(r, i)
    d = te.open_drawer_press(ep, which) if which == "top" else te.open_middle_hook(ep)
    ok = chk0(ep.env) if which == "middle" else d > 0.10
    res.append(ok)
    print(f"{which} ep{i}: 연 거리 {100*d:.1f}cm  {'성공' if ok else '실패'}  스텝 {len(ep.log['pos'])}", flush=True)
    ep.env.close()
print(f"== {which}: {sum(res)}/{n}")
