"""High-resolution snapshots of a scripted 3-stage stacking run (process + final stack from two angles)."""

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from expert import ScriptedExpert
from stack_env import BIN_H, BIN_NAMES, TARGETS, StackEnv, sample_pick, sample_scene

FONT = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 26)
env = StackEnv(render=False)
ex = ScriptedExpert(env, np.random.default_rng(5))
home = ex.home_q()
rng = np.random.default_rng(7)
r = mujoco.Renderer(env.m, 600, 800)


def cam(az, el, dist, look):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.azimuth, c.elevation, c.distance = az, el, dist
    c.lookat[:] = look
    return c


MAIN = cam(205, -28, 0.58, [0.17, -0.01, 0.05])


def shot(c=MAIN, label=None):
    r.update_scene(env.d, c)
    im = Image.fromarray(r.render())
    if label:
        d = ImageDraw.Draw(im)
        w = d.textlength(label, font=FONT)
        d.rectangle([8, 8, 24 + w, 48], fill=(255, 255, 255))
        d.text((16, 10), label, font=FONT, fill=(20, 20, 20))
    return im


shots = []
env.reset(sample_scene(1, rng), home)
shots.append(shot(label="시작: 저울 위의 통 A"))
names = {1: "1층 (통 A)", 2: "옆 (통 B)", 3: "2층 (통 C, A 위)"}
for stage in (1, 2, 3):
    if stage > 1:
        xy, yaw = sample_pick(rng)
        env.spawn_next_bin(stage, xy, yaw)
    bs = env.bin_state(BIN_NAMES[stage - 1])
    tgt = TARGETS[stage] if stage < 3 else env.bin_state("bin_a").pos + np.array([0, 0, BIN_H])
    traj = ex.plan(env.qpos(), bs.pos, bs.yaw, tgt, jitter=False)
    for t, a in enumerate(traj):
        env.step(a)
        if t == 105:
            shots.append(shot(label=f"{names[stage]} 옮기는 중"))
    q0 = env.qpos()
    for i in range(30):
        env.step(q0 + (home - q0) * (i + 1) / 30)
    env.settle(20)
    shots.append(shot(label=f"{names[stage]} 적재 완료"))
    print(stage, env.evaluate_stage(stage)["success"], flush=True)

# process sheet (2 x 4)
W, H = shots[0].size
sheet = Image.new("RGB", (4 * W, 2 * H), "white")
order = [shots[0], shots[1], shots[2], shots[3], shots[4], shots[5], shots[6]]
for i, im in enumerate(order):
    sheet.paste(im, ((i % 4) * W, (i // 4) * H))
final_side = shot(cam(150, -22, 0.42, [0.19, 0.13, 0.05]), "최종: 다른 각도")
sheet.paste(final_side, (3 * W, H))
sheet.resize((2 * W, H)).save("results/stack_process.png")
shot(cam(205, -28, 0.45, [0.19, 0.13, 0.05]), "최종 적재 (1층 A·B, 2층 C)").save("results/stack_final_front.png")
final_side.save("results/stack_final_side.png")
print("saved results/stack_process.png, stack_final_front.png, stack_final_side.png")
r.close()
env.close()
