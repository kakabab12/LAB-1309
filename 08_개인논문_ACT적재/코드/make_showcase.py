"""Large, readable GIFs of trained policies for the journal / README (policy on CPU).

Main view from a free camera that frames the scale and the pallet, the wrist camera as an inset (what the
policy sees), and a caption bar. Modes:
  chained  : one model, stages 1->2->3 on chained eval scene --trial (one GIF per stage + whole sequence)
  compare  : two models side by side on the same scene (chained scene for --stage 1/2, per-stage scene for 3)
usage:
  python make_showcase.py chained --ckpt A1 A2 A3 --label "v5 (최종)" --trial 1 --out results/showcase/v5_seq1
  python make_showcase.py compare --ckpt A1 A2 A3 --ckpt2 B1 B2 B3 --label "v2" --label2 "v5" --stage 1 --trial 1 \
      --out results/showcase/cmp_v2_v5_s1
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from eval_act import EVAL_SEED0, HORIZON, load_policy, to_batch
from expert import ScriptedExpert
from stack_env import BIN_NAMES, StackEnv, sample_pick, sample_scene

W, H = 420, 315
EVERY = 3
FONT = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 17)
SMALL = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 13)
STAGE_KO = {1: "1층", 2: "옆", 3: "2층 (A 위)"}


class Cam:
    def __init__(self, env):
        self.env = env
        self.r = mujoco.Renderer(env.m, H, W)
        self.c = mujoco.MjvCamera()
        self.c.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.c.azimuth, self.c.elevation, self.c.distance = 200, -34, 0.70
        self.c.lookat[:] = [0.17, -0.01, 0.07]

    def frame(self, caption: str, sub: str = "") -> Image.Image:
        self.r.update_scene(self.env.d, self.c)
        im = Image.fromarray(self.r.render())
        wrist = Image.fromarray(self.env.render("front")).resize((128, 96))
        top = H - 122  # empty floor area at the bottom left (scale is mid-left, pallet to the right)
        im.paste(wrist, (6, top + 18))
        d = ImageDraw.Draw(im)
        d.rectangle([5, top + 17, 135, top + 115], outline=(255, 255, 255), width=1)
        d.text((8, top), "손목 카메라", font=SMALL, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        d.rectangle([0, 0, W, 30 if not sub else 48], fill=(255, 255, 255))
        d.text((8, 3), caption, font=FONT, fill=(20, 20, 20))
        if sub:
            d.text((8, 27), sub, font=SMALL, fill=(80, 80, 80))
        return im

    def close(self):
        self.r.close()


def run(env, cam, pol, home, caption, sub_fn):
    """One stage: policy steps then RETURN_HOME, frames every EVERY steps."""
    frames = []
    pol.reset()
    obs = env.observe()
    for t in range(HORIZON):
        x = obs
        if getattr(pol, "preprocess", None):
            import preprocess
            x = {**obs, **{c: preprocess.apply(pol.preprocess, obs[c]) for c in ("front", "top")}}
        with torch.inference_mode():
            a = pol.select_action(to_batch(x, "cpu")).squeeze(0).numpy()
        env.step(a)
        obs = env.observe()
        if t % EVERY == 0:
            frames.append(cam.frame(caption, sub_fn(None)))
    q0 = env.qpos()
    for i in range(30):
        env.step(q0 + (home - q0) * (i + 1) / 30)
        if i % EVERY == 0:
            frames.append(cam.frame(caption, sub_fn(None)))
    env.settle(15)
    return frames


def save(frames, path: Path, hold: int = 8):
    path.parent.mkdir(parents=True, exist_ok=True)
    fr = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames + [frames[-1]] * hold]
    fr[0].save(path, save_all=True, append_images=fr[1:], duration=100, loop=0, optimize=True)
    print("saved", path, f"{path.stat().st_size / 1e6:.1f} MB", flush=True)


def scene_for(stage, trial, chained):
    if chained or stage < 3:
        rng = np.random.default_rng(EVAL_SEED0 + 90000 + trial)
        return rng, sample_scene(1, rng)
    rng = np.random.default_rng(EVAL_SEED0 + 1000 * stage + trial)
    return rng, sample_scene(stage, rng)


def play(env, cam, pols, home, label, stage_from, stage_to, trial, chained):
    """Run stages stage_from..stage_to; earlier chained stages are run without recording."""
    out = {}
    rng, spec = scene_for(stage_to, trial, chained)
    env.reset(spec, home)
    first = 1 if chained else stage_to
    for stage in range(first, stage_to + 1):
        if chained and stage > 1:
            xy, yaw = sample_pick(rng)
            env.spawn_next_bin(stage, xy, yaw)
        refs = {n: env.bin_state(n).pos.copy() for n in BIN_NAMES[: stage - 1]}
        cap = f"{label} · {STAGE_KO[stage]} 적재"
        if stage >= stage_from:
            frames = run(env, cam, pols[stage - 1], home, cap, lambda _: "")
        else:
            frames = run(env, cam, pols[stage - 1], home, cap, lambda _: "")
            frames = []
        r = env.evaluate_stage(stage, ref_positions=refs)
        res = "성공" if r["success"] else f"실패 (위치 {r['xy_err_mm']:.0f}mm, 기울기 {r['tilt_deg']:.0f}°)"
        if frames:
            frames[-1] = cam.frame(cap, f"결과: {res}")
            frames += [frames[-1]] * 6
        out[stage] = (frames, res)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["chained", "compare"])
    ap.add_argument("--ckpt", nargs=3, required=True)
    ap.add_argument("--ckpt2", nargs=3)
    ap.add_argument("--label", required=True)
    ap.add_argument("--label2")
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(2)
    env = StackEnv(render=True)
    home = ScriptedExpert(env).home_q()
    cam = Cam(env)
    out = Path(a.out)
    pols = [load_policy(p, "cpu", None) for p in a.ckpt]
    if a.mode == "chained":
        res = play(env, cam, pols, home, a.label, 1, 3, a.trial, chained=True)
        allf = []
        for st in (1, 2, 3):
            save(res[st][0], out.with_name(out.name + f"_s{st}.gif"))
            allf += res[st][0][::2]
        save(allf, out.with_name(out.name + "_all.gif"), hold=10)
        print({st: res[st][1] for st in res})
    else:
        pols2 = [load_policy(p, "cpu", None) for p in a.ckpt2]
        chained = a.stage < 3
        r1 = play(env, cam, pols, home, a.label, a.stage, a.stage, a.trial, chained)[a.stage]
        r2 = play(env, cam, pols2, home, a.label2, a.stage, a.stage, a.trial, chained)[a.stage]
        n = max(len(r1[0]), len(r2[0]))
        f1 = r1[0] + [r1[0][-1]] * (n - len(r1[0]))
        f2 = r2[0] + [r2[0][-1]] * (n - len(r2[0]))
        both = []
        for x, y in zip(f1, f2):
            im = Image.new("RGB", (2 * W + 6, H), "white")
            im.paste(x, (0, 0))
            im.paste(y, (W + 6, 0))
            both.append(im)
        save(both, out.with_name(out.name + ".gif"))
        print(a.label, r1[1], "|", a.label2, r2[1])
    cam.close()
    env.close()


if __name__ == "__main__":
    main()
