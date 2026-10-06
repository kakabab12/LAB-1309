"""Render a chained 3-stage scripted-expert rollout to GIF (overview + wrist + top cameras)."""
import sys, numpy as np, mujoco
from PIL import Image
from stack_env import StackEnv, sample_scene, sample_pick, TARGETS, BIN_NAMES, BIN_H
from expert import ScriptedExpert
seed=int(sys.argv[1]) if len(sys.argv)>1 else 3
out=sys.argv[2] if len(sys.argv)>2 else "results/gifs/expert_chain.gif"
env=StackEnv(render=True); ex=ScriptedExpert(env, np.random.default_rng(seed)); home=ex.home_q()
rng=np.random.default_rng(seed)
big=mujoco.Renderer(env.m, 360, 480); small=mujoco.Renderer(env.m, 180, 240)
frames=[]
def snap():
    big.update_scene(env.d,"overview"); a=big.render()
    small.update_scene(env.d,"front"); b=small.render()
    small.update_scene(env.d,"top"); c=small.render()
    frames.append(Image.fromarray(np.concatenate([a, np.concatenate([b,c],0)],1)))
for stage in (1,2,3):
    if stage==1: env.reset(sample_scene(1,rng), home)
    else:
        xy,yaw=sample_pick(rng); env.spawn_next_bin(stage,xy,yaw)
    bs=env.bin_state(BIN_NAMES[stage-1])
    tgt=TARGETS[stage] if stage<3 else env.bin_state("bin_a").pos+np.array([0,0,BIN_H])
    for t,a in enumerate(ex.plan(env.qpos(), bs.pos, bs.yaw, tgt)):
        env.step(a)
        if t%3==0: snap()
    q0=env.qpos()
    for i in range(30):
        env.step(q0+(home-q0)*(i+1)/30)
        if i%3==0: snap()
    env.settle(10); print("stage",stage,env.evaluate_stage(stage)["success"])
frames[0].save(out, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
frames[-1].save(out.replace(".gif","_last.png"))
print("saved", out, len(frames))
big.close(); small.close(); env.close()
