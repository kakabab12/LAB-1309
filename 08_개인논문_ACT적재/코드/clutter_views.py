"""Picture of the clutter-test conditions: the same stage-2 scene under each condition, overview + top camera.

usage: python clutter_views.py [--out results/clutter_views.png]
"""

import argparse

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from clutter_eval import CLUTTER_SEED0, CONDS, VIS_SEED0
from eval_act import EVAL_SEED0
from expert import ScriptedExpert
from stack_env import StackEnv, sample_scene

SHOW = ["none", "sparse", "dense", "decoy", "moving", "all", "all_vis", "tight"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/clutter_views.png")
    ap.add_argument("--trial", type=int, default=3)
    a = ap.parse_args()
    env = StackEnv(render=False, clutter=True)
    home = ScriptedExpert(env).home_q()
    big = mujoco.Renderer(env.m, 300, 400)
    small = mujoco.Renderer(env.m, 300, 400)
    tiles = []
    for cond in SHOW:
        n, margin, decoy, mover, vis = CONDS[cond]
        seed = 1000 * 2 + a.trial
        env.reset(sample_scene(2, np.random.default_rng(EVAL_SEED0 + seed)), home)
        if vis:
            env.randomize(np.random.default_rng(VIS_SEED0 + seed))
        else:
            env.reset_visuals()
        env.place_clutter(np.random.default_rng(CLUTTER_SEED0 + seed), n, margin, decoy, mover=mover)
        for _ in range(20):  # let the passing object move a little
            env.step(env.d.ctrl[env.act_ids].copy())
        big.update_scene(env.d, "overview")
        small.update_scene(env.d, "top")
        im = Image.fromarray(np.concatenate([big.render(), small.render()], 1))
        ImageDraw.Draw(im).rectangle((0, 0, 400, 24), fill=(255, 255, 255))
        ImageDraw.Draw(im).text((6, 5), f"{cond}: {n} objects, margin {margin * 1000:.0f} mm"
                                + (", look-alike" if decoy else "") + (", passing" if mover else "")
                                + (", random looks" if vis else ""), fill=(0, 0, 0))
        tiles.append(np.asarray(im))
    rows = [np.concatenate(tiles[i:i + 2], 1) for i in range(0, len(tiles), 2)]
    Image.fromarray(np.concatenate(rows, 0)).save(a.out)
    print(a.out)


if __name__ == "__main__":
    main()
