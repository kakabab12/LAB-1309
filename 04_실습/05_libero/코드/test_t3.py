import sys
import sim_only as so, switch_experiment as sx, task_experts as te
n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
r = so.SimRunner(3); ok_n = 0
for i in range(n):
    ep = sx.Episode(r, i)
    c0 = te.region_pos(ep, "wooden_cabinet_1_top_region")
    te.drawer_bowl(ep, pull=float(sys.argv[2]) if len(sys.argv) > 2 else 0.10, y_front=float(sys.argv[3]) if len(sys.argv) > 3 else 0.05)
    ok = r.chk_a(ep.env); ok_n += ok
    b = ep.obj_pos("akita_black_bowl_1"); c = te.region_pos(ep, "wooden_cabinet_1_top_region")
    print(f"T3 ep{i}: {'성공' if ok else '실패'}  서랍 {100*-te.drawer_qpos(ep,'top'):.1f}cm  그릇 {b.round(3)}  영역 {c.round(3)} (처음 {c0.round(3)})  스텝 {len(ep.log['pos'])}", flush=True)
    ep.env.close()
print(f"== T3: {ok_n}/{n}")
