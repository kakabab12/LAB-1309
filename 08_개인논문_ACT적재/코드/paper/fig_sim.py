"""Figure 2: the simulated cell (overview + the two policy cameras)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from expert import ScriptedExpert
from stack_env import SceneSpec, StackEnv, T1, T2

FONT = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 22)

env = StackEnv(render=True, img_hw=(240, 320))
home = ScriptedExpert(env).home_q()
spec = SceneSpec(stage=3, new_bin_xy=np.array([0.205, -0.140]), new_bin_yaw=0.15,
                 placed={"bin_a": (T1.copy(), 0.0), "bin_b": (T2.copy(), 0.0)})
env.reset(spec, home)
env.settle(30)

big = mujoco.Renderer(env.m, 480, 640)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = [0.15, -0.01, 0.04]
cam.distance = 0.62
cam.azimuth = 205
cam.elevation = -32
big.update_scene(env.d, cam)
over = big.render()
front = env.render("front")
top = env.render("top")


def label(img: np.ndarray, text: str) -> Image.Image:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([6, 6, 6 + 18 * len(text) + 10, 38], fill=(255, 255, 255))
    d.text((12, 8), text, font=FONT, fill=(20, 20, 20))
    return im


a = label(over, "(a)")
b = label(front, "(b)")
c = label(top, "(c)")
W = 640 + 320 + 8
canvas = Image.new("RGB", (W, 480), "white")
canvas.paste(a, (0, 0))
canvas.paste(b, (648, 0))
canvas.paste(c, (648, 240))
canvas.save("paper/fig2_sim.png")
print("saved paper/fig2_sim.png", canvas.size)
big.close()
env.close()
