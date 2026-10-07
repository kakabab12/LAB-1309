"""Why do stage-3 (bin C on top of A) runs end tilted? Runs on CPU next to the training jobs.

1. Expert on the 50 per-stage eval scenes of stage 3: final tilt / position (how close is the expert to 10 deg?).
2. The stage-3 policy on chosen failing trials: bin C height/tilt over time, gripper command, and
   side-view frames at release, after RETURN_HOME and at the end.
usage: ACT_DESIGN=v4 python inspect_stage3.py --ckpt runs/s3_v4_n100/ckpt_030000 --trials 19 20 26
"""

import argparse
import json
import math

import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from eval_act import EVAL_SEED0, HORIZON, load_policy, to_batch
from expert import ScriptedExpert
from stack_env import BIN_H, BIN_NAMES, StackEnv, sample_scene

FONT = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 18)


def tilt_deg(env, name):
    R = env.d.xmat[env.m.body(name).id].reshape(3, 3)
    return math.degrees(math.acos(np.clip(R[2, 2], -1, 1)))


def side_cam(env):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    a = env.d.xpos[env.m.body("bin_a").id]
    c.azimuth, c.elevation, c.distance = 90, -10, 0.30  # looking along +y, sees the x-z plane of the stack
    c.lookat[:] = a + np.array([0, 0, 0.03])
    return c


def side_cam2(env):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    a = env.d.xpos[env.m.body("bin_a").id]
    c.azimuth, c.elevation, c.distance = 180, -10, 0.30  # looking along -x (from the far side toward the robot)
    c.lookat[:] = a + np.array([0, 0, 0.03])
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--trials", type=int, nargs="+", default=[19, 20])
    ap.add_argument("--expert-n", type=int, default=50)
    a = ap.parse_args()

    env = StackEnv(render=True)
    ex = ScriptedExpert(env, np.random.default_rng(0))
    home = ex.home_q()
    tgt_fn = lambda: env.bin_state("bin_a").pos + np.array([0, 0, BIN_H])  # noqa: E731

    # 1. expert on the eval scenes
    ex_rows = []
    for i in range(a.expert_n):
        env.reset(sample_scene(3, np.random.default_rng(EVAL_SEED0 + 3000 + i)), home)
        bs = env.bin_state(BIN_NAMES[2])
        traj = ex.plan(env.qpos(), bs.pos, bs.yaw, tgt_fn())
        for act in traj:
            env.step(act)
        q0 = env.qpos()
        for k in range(30):
            env.step(q0 + (home - q0) * (k + 1) / 30)
        env.settle(15)
        r = env.evaluate_stage(3)
        ex_rows.append({"trial": i, "success": bool(r["success"]), "tilt": r["tilt_deg"], "xy": r["xy_err_mm"]})
    tl = np.array([x["tilt"] for x in ex_rows])
    print(f"expert stage 3 on eval scenes: success {sum(x['success'] for x in ex_rows)}/{len(ex_rows)}, "
          f"tilt mean {tl.mean():.1f} max {tl.max():.1f} deg, >5 deg: {(tl > 5).sum()}", flush=True)

    # 2. policy on failing trials
    pol = load_policy(a.ckpt, "cpu", None)
    r = mujoco.Renderer(env.m, 300, 400)
    sheets, logs = [], {}
    for i in a.trials:
        env.reset(sample_scene(3, np.random.default_rng(EVAL_SEED0 + 3000 + i)), home)
        pol.reset()
        obs = env.observe()
        frames, log, released = [], [], None
        for t in range(HORIZON):
            with torch.inference_mode():
                act = pol.select_action(to_batch(obs, "cpu")).squeeze(0).numpy()
            env.step(act)
            obs = env.observe()
            c = env.d.xpos[env.m.body("bin_c").id]
            log.append([t, float(act[5]), float(1000 * c[2]), tilt_deg(env, "bin_c")])
            if released is None and t > 120 and act[5] > 0.0:
                released = t
                for cam in (side_cam(env), side_cam2(env)):
                    r.update_scene(env.d, cam)
                    frames.append((f"t{i} 놓는 순간 (t={t})", Image.fromarray(r.render())))
        q0 = env.qpos()
        for k in range(30):
            env.step(q0 + (home - q0) * (k + 1) / 30)
        env.settle(15)
        res = env.evaluate_stage(3)
        for cam in (side_cam(env), side_cam2(env)):
            r.update_scene(env.d, cam)
            frames.append((f"t{i} 끝 (기울기 {res['tilt_deg']:.1f}°)", Image.fromarray(r.render())))
        lg = np.array(log)
        # tilt just before/after release
        tb = lg[released - 1, 3] if released else float("nan")
        ta = lg[min(len(lg) - 1, released + 20), 3] if released else float("nan")
        print(f"policy trial {i}: success {res['success']} tilt_end {res['tilt_deg']:.1f} xy {res['xy_err_mm']:.1f} "
              f"| release t={released}, tilt before {tb:.1f} after {ta:.1f}, C height at release "
              f"{lg[released - 1, 2] if released else float('nan'):.1f} mm", flush=True)
        logs[i] = {"release_t": released, "result": {k: v for k, v in res.items() if k != "disturbed"},
                   "log": lg.tolist()}
        sheets.append(frames)
    W, H = 400, 300
    sheet = Image.new("RGB", (W * 4, H * len(sheets)), "white")
    for row, frames in enumerate(sheets):
        for col, (lab, im) in enumerate(frames[:4]):
            d = ImageDraw.Draw(im)
            d.rectangle([4, 4, 12 + d.textlength(lab, font=FONT), 30], fill=(255, 255, 255))
            d.text((8, 4), lab, font=FONT, fill=(20, 20, 20))
            sheet.paste(im, (col * W, row * H))
    sheet.save("results/stage3_inspect.png")
    json.dump({"expert": ex_rows, "policy": logs}, open("results/stage3_inspect.json", "w"))
    print("saved results/stage3_inspect.png")
    r.close()
    env.close()


if __name__ == "__main__":
    main()
