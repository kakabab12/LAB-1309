"""Figure 4: frames of a chained ACT rollout (policies only, no scripted help).

usage: python paper/fig_rollout.py --ckpt runs/s1_X/ckpt_030000 runs/s2_X/ckpt_030000 runs/s3_X/ckpt_030000
Tries evaluation seeds until a full 3-stage success is found (the seed is printed and saved).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from eval_act import EVAL_SEED0, HORIZON, load_policy, to_batch
from expert import ScriptedExpert
from stack_env import BIN_NAMES, StackEnv, sample_pick, sample_scene

FONT = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 26)
LABELS = ["(a) 대기", "(b) 1층 집기", "(c) 1층 적재", "(d) 옆 적재", "(e) 2층 집기", "(f) 2층 적재"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs=3, required=True)
    ap.add_argument("--max-tries", type=int, default=15)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "fig4_rollout.png"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = StackEnv(render=True)
    home = ScriptedExpert(env).home_q()
    pols = [load_policy(p, device, None) for p in args.ckpt]
    r = mujoco.Renderer(env.m, 360, 480)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.17, -0.01, 0.05]
    cam.distance = 0.55
    cam.azimuth = 205
    cam.elevation = -30

    def shot():
        r.update_scene(env.d, cam)
        return r.render().copy()

    for k in range(args.max_tries):
        seed = EVAL_SEED0 + 90000 + k
        rng = np.random.default_rng(seed)
        env.reset(sample_scene(1, rng), home)
        frames = [shot()]
        ok_all = True
        for stage in (1, 2, 3):
            if stage > 1:
                xy, yaw = sample_pick(rng)
                env.spawn_next_bin(stage, xy, yaw)
            refs = {n: env.bin_state(n).pos.copy() for n in BIN_NAMES[: stage - 1]}
            pol = pols[stage - 1]
            pol.reset()
            obs = env.observe()
            opened, grabbed = False, False
            for t in range(HORIZON):
                with torch.inference_mode():
                    a = pol.select_action(to_batch(obs, device)).squeeze(0).cpu().numpy()
                env.step(a)
                obs = env.observe()
                opened = opened or a[5] > 0.2
                if not grabbed and opened and a[5] < -0.1:
                    grabbed = True
                    if stage in (1, 3):
                        for _ in range(12):  # let the jaws close before the snapshot
                            env.step(a)
                        obs = env.observe()
                        frames.append(shot())
            q0 = env.qpos()
            for i in range(30):
                env.step(q0 + (home - q0) * (i + 1) / 30)
            env.settle(15)
            res = env.evaluate_stage(stage, ref_positions=refs)
            ok_all = ok_all and res["success"]
            frames.append(shot())
        print(f"seed {seed}: success={ok_all}, frames={len(frames)}")
        if ok_all and len(frames) == 6:
            break
    else:
        print("no fully successful chained rollout found; using the last one")

    tiles = []
    for img, lab in zip(frames, LABELS):
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        w = d.textlength(lab, font=FONT)
        d.rectangle([6, 6, 18 + w, 44], fill=(255, 255, 255))
        d.text((12, 8), lab, font=FONT, fill=(20, 20, 20))
        tiles.append(im)
    W, H = tiles[0].size
    gap = 6
    canvas = Image.new("RGB", (3 * W + 2 * gap, 2 * H + gap), "white")
    for i, im in enumerate(tiles):
        canvas.paste(im, ((i % 3) * (W + gap), (i // 3) * (H + gap)))
    canvas.save(args.out)
    Path(args.out).with_suffix(".json").write_text(json.dumps({"seed": seed, "success": ok_all, "ckpt": args.ckpt}))
    print("saved", args.out)
    r.close()
    env.close()


if __name__ == "__main__":
    main()
