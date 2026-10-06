import sys, numpy as np
from stack_env import StackEnv, sample_scene, sample_pick, TARGETS, BIN_NAMES, BIN_H
from expert import ScriptedExpert
env = StackEnv(render=False); ex = ScriptedExpert(env, np.random.default_rng(7)); home = ex.home_q()
N=int(sys.argv[1]) if len(sys.argv)>1 else 10
rng=np.random.default_rng(123)
res={1:[],2:[],3:[]}; chain=0; ikmax=0
for ep in range(N):
    refs={}; ok_chain=True
    for stage in (1,2,3):
        if stage==1: env.reset(sample_scene(1,rng), home)
        else:
            xy,yaw=sample_pick(rng); env.spawn_next_bin(stage,xy,yaw)
        for n in BIN_NAMES[:stage-1]: refs[n]=env.bin_state(n).pos.copy()
        bs=env.bin_state(BIN_NAMES[stage-1])
        tgt=TARGETS[stage] if stage<3 else env.bin_state("bin_a").pos+np.array([0,0,BIN_H])
        for a in ex.plan(env.qpos(), bs.pos, bs.yaw, tgt): env.step(a)
        ikmax=max(ikmax, ex.last_ik_err)
        q0=env.qpos()
        for i in range(30): env.step(q0+(home-q0)*(i+1)/30)
        env.settle(20)
        r=env.evaluate_stage(stage, ref_positions=refs); res[stage].append(r)
        ok_chain = ok_chain and r["success"]
    chain+=ok_chain
for s in (1,2,3):
    rs=res[s]; print(f"stage {s}: success {sum(r['success'] for r in rs)}/{len(rs)}  xy_err mean {np.mean([r['xy_err_mm'] for r in rs]):.1f} mm  max {np.max([r['xy_err_mm'] for r in rs]):.1f}  fails {[ (round(r['xy_err_mm']),round(r['z_err_mm']),round(r['tilt_deg']),r['disturbed']) for r in rs if not r['success']][:5]}")
print("chain", chain, "/", N, " max IK err mm", round(ikmax*1000,2))
