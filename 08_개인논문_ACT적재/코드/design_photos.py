"""Photos of the grasp moment for each demonstration design (v1, v2, v3b, v4), for the journal.

For each design the scripted expert runs until the moment it starts closing the gripper; two close-ups are
taken: from above (which wall is grasped) and along the wall (how far the fixed jaw is from the wall).
v2 and v4 are also shown with the bin rotated +15 deg, where the old robot-facing rule switches walls.
"""

import math

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import expert as E
from stack_env import BIN_NAMES, SCALE_C, TARGETS, StackEnv, sample_scene

FONT = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 22)
SMALL = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 18)
W, H = 420, 330
FIXED = np.array([-1.0, 0.0])
CASES = [  # title, subtitle, clearance, opening, wall rule (None = robot-facing), bin yaw (deg)
    ("v1", "여유 1.5mm · 로봇 쪽 벽 · 회전 0°", 0.0015, 0.32, None, 0),
    ("v2", "여유 9mm · 로봇 쪽 벽 · 회전 0°", 0.009, 0.32, None, 0),
    ("v3b (시험만)", "여유 13mm · 열림 0.42 · 회전 0°", 0.013, 0.42, None, 0),
    ("v2, 통 +15°", "로봇 쪽 벽 규칙 → 옆벽을 잡음", 0.009, 0.32, None, 15),
    ("v4, 통 +15°", "늘 같은 벽 → 앞벽을 잡음", 0.009, 0.32, FIXED, 15),
]

env = StackEnv(render=False)
r = mujoco.Renderer(env.m, H, W)
BIN_GEOMS = [g for g in range(env.m.ngeom) if env.m.geom_bodyid[g] == env.m.body(BIN_NAMES[0]).id]
OPAQUE = env.m.geom_rgba[BIN_GEOMS].copy()


def see_through(on: bool) -> None:  # translucent bin walls so the jaws behind them are visible
    env.m.geom_rgba[BIN_GEOMS] = OPAQUE
    if on:
        env.m.geom_rgba[BIN_GEOMS, 3] = 0.35


def cam(az, el, dist, look):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.azimuth, c.elevation, c.distance = az, el, dist
    c.lookat[:] = look
    return c


def shot(c, title=None, sub=None):
    r.update_scene(env.d, c)
    im = Image.fromarray(r.render())
    d = ImageDraw.Draw(im)
    if title:
        w = max(d.textlength(title, font=FONT), d.textlength(sub or "", font=SMALL))
        d.rectangle([6, 6, 20 + w, 64 if sub else 40], fill=(255, 255, 255))
        d.text((12, 6), title, font=FONT, fill=(20, 20, 20))
        if sub:
            d.text((12, 36), sub, font=SMALL, fill=(70, 70, 70))
    return im


tops, sides = [], []
for title, sub, clear, opening, rule, yaw_d in CASES:
    E.GRIPPER_OPEN = opening  # plan() reads the module constants
    E.WALL_REF = rule
    ex = E.ScriptedExpert(env, np.random.default_rng(0))
    ex.CLEAR = clear
    home = ex.home_q()
    spec = sample_scene(1, np.random.default_rng(1))
    spec.new_bin_xy = SCALE_C.copy()
    spec.new_bin_yaw = math.radians(yaw_d)
    env.reset(spec, home)
    bs = env.bin_state(BIN_NAMES[0])
    n = ex.facing_wall_normal(bs.pos[:2], bs.yaw)
    traj = ex.plan(env.qpos(), bs.pos, bs.yaw, TARGETS[1], jitter=False)
    opened = False
    for a in traj:
        opened |= a[5] > 0.1
        if opened and a[5] < 0.0:
            break  # jaws open at grasp depth, about to close
        env.step(a)
    wall = bs.pos + np.array([*(0.032 * n[:2]), 0.012])
    # from behind the bin, looking toward the robot (camera forward = robot direction from the scale)
    to_robot = -SCALE_C / np.linalg.norm(SCALE_C)
    see_through(False)
    tops.append(shot(cam(math.degrees(math.atan2(to_robot[1], to_robot[0])), -50, 0.30, bs.pos + [0, 0, 0.02]),
                     title, sub))
    t = np.array([-n[1], n[0]])
    see_through(True)
    sides.append(shot(cam(math.degrees(math.atan2(t[1], t[0])), -4, 0.095, wall), "벽을 따라 본 모습 (통 반투명)"))
    print(title, "normal", np.round(n[:2], 2), flush=True)

sheet = Image.new("RGB", (len(CASES) * W, 2 * H), "white")
for i, (a, b) in enumerate(zip(tops, sides)):
    sheet.paste(a, (i * W, 0))
    sheet.paste(b, (i * W, H))
sheet.save("results/design_versions.png")
print("saved results/design_versions.png")
r.close()
env.close()
