"""Pictures of random factory-cell scenes (factory.py): overview camera, top camera and wrist camera at the start of a
stage, several seeds. Also reports how often the scripted expert completes the stage without touching anything.

usage: [ACT_OBJECT=cup ACT_LAYOUT=mirror ACT_DESIGN=cuprule] python factory_views.py --n 6 --out results/factory_views.png
"""

import argparse

import mujoco
import numpy as np
from PIL import Image, ImageDraw

import factory
from expert import ScriptedExpert, stage2_target
from stack_env import BIN_H, BIN_NAMES, TARGETS, StackEnv, sample_scene


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--expert", type=int, default=0, help="also run the expert on this many scenes per stage")
    ap.add_argument("--out", default="results/factory_views.png")
    a = ap.parse_args()
    env = StackEnv(render=True, dr=True, factory=True)
    ex = ScriptedExpert(env, np.random.default_rng(0))
    home = ex.home_q()
    big = mujoco.Renderer(env.m, 300, 400)
    small = mujoco.Renderer(env.m, 150, 200)
    tiles = []
    for i in range(a.n):
        rng = np.random.default_rng(a.seed + i)
        env.reset(sample_scene(a.stage, rng, wide=True), home)
        env.randomize(rng, level=1 + i % 2)
        info = factory.place_factory(env, rng, 14, 0.010)
        for _ in range(15):
            env.step(env.d.ctrl[env.act_ids].copy())
        big.update_scene(env.d, "overview")
        o = big.render()
        small.update_scene(env.d, "top")
        t = small.render()
        small.update_scene(env.d, "front")
        f = small.render()
        im = Image.fromarray(np.concatenate([o, np.concatenate([t, f], 0)], 1))
        n_obj = len([k for k in info["clutter"] if k != "mover"])
        txt = (f"conveyor {'Y' if info['conveyor'] else '-'} rack {'Y' if info['rack'] else '-'} "
               f"ctrl {'Y' if info['ctrlbox'] else '-'} posts {sum(p is not None for p in info['pillars'])} "
               f"cables {len(info['cables'])} tapes {len(info['tapes'])} parts {n_obj}")
        ImageDraw.Draw(im).rectangle((0, 0, 400, 16), fill=(255, 255, 255))
        ImageDraw.Draw(im).text((4, 2), txt, fill=(0, 0, 0))
        tiles.append(np.asarray(im))
    rows = [np.concatenate(tiles[i:i + 2], 1) for i in range(0, len(tiles) - 1, 2)]
    Image.fromarray(np.concatenate(rows, 0)).save(a.out)
    print(a.out)
    for stage in (1, 2, 3) if a.expert else ():
        ok = 0
        for i in range(a.expert):
            rng = np.random.default_rng(9000 + 100 * stage + i)
            env.reset(sample_scene(stage, rng, wide=True), home)
            env.randomize(rng, level=1 + i % 2)
            factory.place_factory(env, rng, int(rng.integers(8, 17)), float(rng.uniform(0.005, 0.020)))
            refs = {b: env.bin_state(b).pos.copy() for b in BIN_NAMES[: stage - 1]}
            crefs = env.clutter_positions()
            bs = env.bin_state(BIN_NAMES[stage - 1])
            a_now = env.bin_state("bin_a").pos
            tgt = {1: TARGETS[1], 2: stage2_target(a_now), 3: a_now + np.array([0, 0, BIN_H])}[stage]
            for act in ex.plan(env.qpos(), bs.pos, bs.yaw, tgt):
                env.step(act)
            env.settle(10)
            r = env.evaluate_stage(stage, ref_positions=refs)
            moved = env.clutter_moved(crefs)
            ok += r["success"] and not moved and not env.struct_hit
            if not (r["success"] and not moved and not env.struct_hit):
                print(f"  stage {stage} scene {i}: xy {r['xy_err_mm']:.1f} moved {moved} struct_hit {env.struct_hit}")
        print(f"expert stage {stage}: {ok}/{a.expert}", flush=True)


if __name__ == "__main__":
    main()
