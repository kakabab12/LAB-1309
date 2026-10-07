"""Does the gripper close firmly on the bin wall? Measures, for expert stage-1 runs:

- jaw angle after closing vs the closed command (the 2 mm wall stops the jaws before the command),
- gripper servo torque (saturated = squeezing at full force) and pad-wall contact normal force,
- slip of the bin inside the gripper while carried (position and tilt change in the gripper frame).
Also saves a photo pair: just closed (wall translucent) and mid-carry.
Run with ACT_DESIGN=v4 (or v2).
"""

import json
import math
import os

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import expert as E
from stack_env import BIN_NAMES, GRIPPER_CLOSED, GRIPPER_OPEN, SCALE_C, TARGETS, StackEnv, sample_scene

env = StackEnv(render=False)
m, d = env.m, env.d
ex = E.ScriptedExpert(env, np.random.default_rng(0))
home = ex.home_q()
GJ = m.joint("gripper").id
GQ = m.jnt_qposadr[GJ]
GA = m.actuator("gripper").id
PADS = {m.geom("fixed_jaw_pad").id: "fixed", m.geom("moving_jaw_pad").id: "moving"}
BIN = m.body(BIN_NAMES[0]).id
r = mujoco.Renderer(m, 330, 420)
FONT = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 20)
BIN_GEOMS = [g for g in range(m.ngeom) if m.geom_bodyid[g] == BIN]
OPAQUE = m.geom_rgba[BIN_GEOMS].copy()


def pad_forces():
    f = {"fixed": 0.0, "moving": 0.0, "pen_mm": 0.0}
    c6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            if a in PADS and m.geom_bodyid[b] == BIN:
                mujoco.mj_contactForce(m, d, i, c6)
                f[PADS[a]] += abs(c6[0])
                f["pen_mm"] = max(f["pen_mm"], -1000 * c.dist)  # pad-wall overlap of the soft contact
    return f


def rel_pose():
    """Bin centre and up-axis expressed in the gripper-site frame."""
    R = d.site_xmat[env.site].reshape(3, 3)
    p = R.T @ (d.xpos[BIN] - d.site_xpos[env.site])
    up = R.T @ d.xmat[BIN].reshape(3, 3)[:, 2]
    return p, up


def photo(title, see_through, n):
    m.geom_rgba[BIN_GEOMS] = OPAQUE
    if see_through:
        m.geom_rgba[BIN_GEOMS, 3] = 0.35
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    t = np.array([-n[1], n[0]])
    c.azimuth, c.elevation, c.distance = math.degrees(math.atan2(t[1], t[0])), -4, 0.11
    c.lookat[:] = d.xpos[BIN] + np.array([*(0.032 * n[:2]), 0.012])
    r.update_scene(d, c)
    im = Image.fromarray(r.render())
    dr = ImageDraw.Draw(im)
    dr.rectangle([6, 6, 18 + dr.textlength(title, font=FONT), 36], fill=(255, 255, 255))
    dr.text((12, 6), title, font=FONT, fill=(20, 20, 20))
    m.geom_rgba[BIN_GEOMS] = OPAQUE
    return im


rows, pics = [], []
for yaw_d in (0, 15, -15):
    for corner in ((0, 0), (1, -1), (-1, 1)):
        spec = sample_scene(1, np.random.default_rng(3))
        spec.new_bin_xy = SCALE_C + 0.02 * np.array(corner)
        spec.new_bin_yaw = math.radians(yaw_d)
        env.reset(spec, home)
        bs = env.bin_state(BIN_NAMES[0])
        n = ex.facing_wall_normal(bs.pos[:2], bs.yaw)
        traj = ex.plan(env.qpos(), bs.pos, bs.yaw, TARGETS[1], jitter=False)
        closing, closed_at, ref, out, last = False, None, None, {}, None
        for t, a in enumerate(traj):
            env.step(a)
            if not closing and a[5] < GRIPPER_OPEN - 0.01 and t > 30:
                closing = True
            if closing and closed_at is None and a[5] <= GRIPPER_CLOSED + 1e-3:
                closed_at = t
            if closed_at is not None and t == closed_at + 5:  # closed for 5 steps, still at grasp height
                f = pad_forces()
                out.update(jaw_q=float(d.qpos[GQ]), cmd=float(d.ctrl[GA]), torque=float(d.actuator_force[GA]),
                           f_fixed=f["fixed"], f_moving=f["moving"], pen_mm=f["pen_mm"])
                ref = rel_pose()
                if yaw_d == 0 and corner == (0, 0):
                    pics.append(photo("닫은 직후 (통 반투명)", True, n))
            if ref is not None and "slip_mm" not in out:
                if a[5] > GRIPPER_CLOSED + 0.01:  # release command started: use the state one step before
                    p, up, f = last
                    out["slip_mm"] = float(1000 * np.linalg.norm(p - ref[0]))
                    out["tilt_change_deg"] = float(math.degrees(math.acos(np.clip(up @ ref[1], -1, 1))))
                    out["f_before_release"] = f["fixed"] + f["moving"]
                else:
                    last = (*rel_pose(), pad_forces())
            if ref is not None and t == closed_at + 45:
                f = pad_forces()
                out["f_carry"] = f["fixed"] + f["moving"]
                out["jaw_q_carry"] = float(d.qpos[GQ])
            if ref is not None and yaw_d == 0 and corner == (0, 0) and t == closed_at + 46:
                pics.append(photo("들어 올려 옮기는 중", False, n))
        env.settle(15)
        out.update(yaw=yaw_d, corner=list(corner), success=bool(env.evaluate_stage(1)["success"]))
        rows.append(out)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}, flush=True)

design = os.environ.get("ACT_DESIGN", "v2")
json.dump(rows, open(f"results/grip_check_{design}.json", "w"), indent=1)
if pics:
    sheet = Image.new("RGB", (420 * len(pics), 330), "white")
    for i, im in enumerate(pics):
        sheet.paste(im, (420 * i, 0))
    sheet.save(f"results/grip_check_{design}.png")
print("jaw angle when closed on the wall (rad): mean", np.mean([x["jaw_q"] for x in rows]),
      "| fully closed command", GRIPPER_CLOSED)
r.close()
env.close()
