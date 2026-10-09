"""Picture of the generalisation objects: the scripted expert stacks 1st floor -> beside -> on top for one object /
layout (rule design), and the overview camera shows the scene before and after. Run once per object / layout
(ACT_OBJECT / ACT_LAYOUT / ACT_DESIGN), then --merge puts the tiles together.

usage: ACT_OBJECT=cup ACT_DESIGN=cuprule python object_views.py --tile results/obj_tiles/cup_orig.png
       python object_views.py --merge results/obj_tiles/*.png --out results/object_views.png
"""

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw


def tile(path: str) -> None:
    import mujoco

    from expert import ScriptedExpert, stage2_target
    from stack_env import BIN_H, BIN_NAMES, TARGETS, StackEnv, sample_pick, sample_scene

    env = StackEnv(render=False)
    rng = np.random.default_rng(4)
    ex = ScriptedExpert(env, rng)
    home = ex.home_q()
    r = mujoco.Renderer(env.m, 300, 400)
    env.reset(sample_scene(1, rng), home)
    r.update_scene(env.d, "overview")
    before = r.render().copy()
    for stage in (1, 2, 3):
        if stage > 1:
            xy, yaw = sample_pick(rng)
            env.spawn_next_bin(stage, xy, yaw)
        bs = env.bin_state(BIN_NAMES[stage - 1])
        a = env.bin_state("bin_a").pos
        tgt = {1: TARGETS[1], 2: stage2_target(a), 3: a + np.array([0, 0, BIN_H])}[stage]
        for act in ex.plan(env.qpos(), bs.pos, bs.yaw, tgt):
            env.step(act)
        q0 = env.qpos()
        for i in range(30):
            env.step(q0 + (home - q0) * (i + 1) / 30)
        env.settle(15)
    r.update_scene(env.d, "overview")
    after = r.render().copy()
    im = Image.fromarray(np.concatenate([before, after], 1))
    label = f"{os.environ.get('ACT_OBJECT', 'bin')} / {os.environ.get('ACT_LAYOUT', 'orig')} layout: before -> after 3 stages"
    ImageDraw.Draw(im).rectangle((0, 0, 800, 22), fill=(255, 255, 255))
    ImageDraw.Draw(im).text((6, 4), label, fill=(0, 0, 0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile")
    ap.add_argument("--merge", nargs="+")
    ap.add_argument("--out", default="results/object_views.png")
    a = ap.parse_args()
    if a.tile:
        tile(a.tile)
    if a.merge:
        ims = [np.asarray(Image.open(p).convert("RGB")) for p in a.merge]
        rows = [np.concatenate(ims[i:i + 2], 1) if i + 1 < len(ims) else np.concatenate([ims[i], np.full_like(ims[i], 255)], 1)
                for i in range(0, len(ims), 2)]
        Image.fromarray(np.concatenate(rows, 0)).save(a.out)
        print(a.out)


if __name__ == "__main__":
    main()
