"""전환 상황에서 전문가가 B 와 재개를 얼마나 해내나 (모델 없이, 저장 안 함). 모델이 넘을 수 없는 천장."""
import sys
import numpy as np
import collect_expert as ce, scripted_expert as se, sim_only as so, switch_experiment as sx, task_experts as te
import switch_v6 as sv
from collect_v6 import settle
pairs = [tuple(map(int, p.split(":"))) for p in sys.argv[1].split(",")]
n = int(sys.argv[2])
for at, bt in pairs:
    r = so.SimRunner(at, bt); kb = kr = tried = rtry = 0
    for i in range(n):
        ep = sx.Episode(r, i); obj = sv.OBJ[at]; z = ep.obj_pos(obj)[2]
        if not sv.run_until_held(ep, at, ce.Rec()):
            ep.env.close(); continue
        tried += 1; t_sw = len(ep.log["pos"])
        sv.do_b(ep, at, bt, ce.Rec(), z)
        ok = settle(ep, r.chk_b) and se.min_home_dist(ep, t_sw) >= 0.07
        kb += ok
        if ok and sv.OBJ.get(bt) != obj and not r.chk_a(ep.env):
            rtry += 1
            te.EXPERT[at](ep, None)
            kr += settle(ep, r.chk_a) and bool(r.chk_b(ep.env))
        ep.env.close()
    print(f"== A{at}→B{bt}: 전환 {kb}/{tried}  재개 {kr}/{rtry}", flush=True)
