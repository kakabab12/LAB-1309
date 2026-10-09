"""GIF of the scripted expert stacking all three stages in one scene (overview + wrist + top camera), optionally in
a factory cell (factory.py) with randomised looks: what the demonstrations of the generalisation / integrated tests
look like. Object, layout and design come from ACT_OBJECT / ACT_LAYOUT / ACT_DESIGN.

usage: ACT_OBJECT=cup ACT_DESIGN=cuprule python expert_gif.py --factory --seed 3 --out results/gifs_expert/cup_factory.gif
"""

import argparse
from pathlib import Path

import numpy as np

from eval_act import Recorder
from expert import ScriptedExpert, stage2_target
from stack_env import BIN_H, BIN_NAMES, TARGETS, StackEnv, sample_pick, sample_scene


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factory", action="store_true")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    env = StackEnv(render=True, dr=a.factory, factory=a.factory)
    rng = np.random.default_rng(a.seed)
    ex = ScriptedExpert(env, rng)
    home = ex.home_q()
    rec = Recorder(env, enabled=True)
    env.reset(sample_scene(1, rng), home)
    if a.factory:
        import factory
        env.randomize(rng, level=1 + a.seed % 2)
        factory.place_factory(env, rng, 14, 0.010)
    ok = []
    for stage in (1, 2, 3):
        if stage > 1:
            xy, yaw = sample_pick(rng)
            env.spawn_next_bin(stage, xy, yaw)
        refs = {b: env.bin_state(b).pos.copy() for b in BIN_NAMES[: stage - 1]}
        bs = env.bin_state(BIN_NAMES[stage - 1])
        an = env.bin_state("bin_a").pos
        tgt = {1: TARGETS[1], 2: stage2_target(an), 3: an + np.array([0, 0, BIN_H])}[stage]
        for t, act in enumerate(ex.plan(env.qpos(), bs.pos, bs.yaw, tgt)):
            env.step(act)
            if t % 4 == 0:
                rec.snap()
        q0 = env.qpos()
        for i in range(30):
            env.step(q0 + (home - q0) * (i + 1) / 30)
            if i % 4 == 0:
                rec.snap()
        env.settle(10)
        ok.append(env.evaluate_stage(stage, ref_positions=refs)["success"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rec.save(Path(a.out))
    print(a.out, ok)


if __name__ == "__main__":
    main()
