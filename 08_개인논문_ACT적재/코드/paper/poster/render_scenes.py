"""High-resolution scene renders for the conference poster (90 x 120 cm print).

The paper figures were rendered at 400 x 300 px; printed 13 cm wide that is ~75 dpi. These are the same scenes,
same scripted expert and cameras, rendered at 1600 x 1200 (the model's offscreen limit).
Object / design come from the environment like the other scripts (stack_env reads them at import):
  ACT_DESIGN=v5 python paper/poster/render_scenes.py basic          # 3 bins stacked + wrist / top camera views
  ACT_OBJECT=cup ACT_DESIGN=cuprule python paper/poster/render_scenes.py object --name cup
  ACT_OBJECT=box ACT_DESIGN=v5 python paper/poster/render_scenes.py object --name box
  python paper/poster/render_scenes.py clutter | factory | looks
  ACT_DESIGN=v4 PALLET_VIA=far python paper/poster/render_scenes.py pallet
"""

import argparse
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
OUT = Path(__file__).resolve().parent / "img"
H, W = 1200, 1600


def save(arr, name):
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(OUT / name)
    print("saved", OUT / name, arr.shape)


def stack_three(env, ex, rng, home):
    from expert import stage2_target
    from stack_env import BIN_H, BIN_NAMES, TARGETS, sample_pick
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


def close_cam():
    from stack_env import TARGETS
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [*(TARGETS[1][:2] + TARGETS[2][:2]) / 2, 0.045]
    cam.distance, cam.azimuth, cam.elevation = 0.36, 205 if TARGETS[2][1] > 0 else 155, -28
    return cam


def basic():
    from expert import ScriptedExpert
    from stack_env import StackEnv, sample_scene
    env = StackEnv(render=False)
    rng = np.random.default_rng(4)
    ex = ScriptedExpert(env, rng)
    home = ex.home_q()
    big = mujoco.Renderer(env.m, H, W)
    cam_r = mujoco.Renderer(env.m, 600, 800)
    env.reset(sample_scene(1, rng), home)
    big.update_scene(env.d, "overview"); save(big.render(), "basic_before.png")
    for c in ("front", "top"):  # what the model sees (rendered larger than its 160 x 120 input, same camera)
        cam_r.update_scene(env.d, c); save(cam_r.render(), f"basic_cam_{c}.png")
    stack_three(env, ex, rng, home)
    big.update_scene(env.d, "overview"); save(big.render(), "basic_after.png")
    big.update_scene(env.d, close_cam()); save(big.render(), "basic_close.png")


def obj(name):
    from expert import ScriptedExpert
    from stack_env import StackEnv, sample_scene
    env = StackEnv(render=False)
    rng = np.random.default_rng(4)
    ex = ScriptedExpert(env, rng)
    home = ex.home_q()
    big = mujoco.Renderer(env.m, H, W)
    env.reset(sample_scene(1, rng), home)
    stack_three(env, ex, rng, home)
    big.update_scene(env.d, "overview"); save(big.render(), f"{name}_after.png")
    big.update_scene(env.d, close_cam()); save(big.render(), f"{name}_close.png")


def clutter():
    from clutter_eval import CLUTTER_SEED0, CONDS
    from eval_act import EVAL_SEED0
    from expert import ScriptedExpert
    from stack_env import StackEnv, sample_scene
    env = StackEnv(render=False, clutter=True)
    home = ScriptedExpert(env).home_q()
    big = mujoco.Renderer(env.m, H, W)
    n, margin, decoy, mover, vis = CONDS["all"]
    seed = 1000 * 2 + 3  # the scene of results/clutter_views.png
    env.reset(sample_scene(2, np.random.default_rng(EVAL_SEED0 + seed)), home)
    env.reset_visuals()
    env.place_clutter(np.random.default_rng(CLUTTER_SEED0 + seed), n, margin, decoy, mover=mover)
    for _ in range(20):
        env.step(env.d.ctrl[env.act_ids].copy())
    big.update_scene(env.d, "overview"); save(big.render(), "clutter.png")


def factory_scene():
    import factory
    from expert import ScriptedExpert
    from stack_env import StackEnv, sample_scene
    env = StackEnv(render=True, dr=True, factory=True)
    home = ScriptedExpert(env, np.random.default_rng(0)).home_q()
    big = mujoco.Renderer(env.m, H, W)
    rng = np.random.default_rng(31)  # first scene of results/factory_views.png
    env.reset(sample_scene(2, rng, wide=True), home)
    env.randomize(rng, level=1)
    factory.place_factory(env, rng, 14, 0.010)
    for _ in range(15):
        env.step(env.d.ctrl[env.act_ids].copy())
    big.update_scene(env.d, "overview"); save(big.render(), "factory.png")


def looks(seeds=(3, 7, 11, 19)):
    from expert import ScriptedExpert
    from stack_env import StackEnv, sample_scene
    env = StackEnv(render=False, dr=True)
    home = ScriptedExpert(env).home_q()
    big = mujoco.Renderer(env.m, H, W)
    for s in seeds:  # a few random looks (lighting, table colour, camera pose): pick one for the poster
        rng = np.random.default_rng(s)
        env.reset(sample_scene(2, rng), home)
        env.randomize(rng, level=1)
        env.settle(5)
        big.update_scene(env.d, "overview"); save(big.render(), f"looks_{s}.png")


def pallet():
    import pallet_grid as PG
    import stack_env as SE
    slots = PG.configure(2, 2, 2, 0.20, 0.03)
    env = PG.make_env(slots, render=False)
    ex = PG.GridExpert(env, np.random.default_rng(11))
    home = ex.home_q()
    rng = np.random.default_rng(2024)
    big = mujoco.Renderer(env.m, H, W)
    xy, yaw = SE.sample_pick(rng)
    env.reset(SE.SceneSpec(stage=1, new_bin_xy=xy, new_bin_yaw=yaw), home)
    ok = []
    for k in range(len(slots)):
        if k > 0:
            xy, yaw = SE.sample_pick(rng)
            env.spawn_next_bin(k + 1, xy, yaw)
        refs = {s["name"]: env.bin_state(s["name"]).pos.copy() for s in slots[:k]}
        bs = env.bin_state(slots[k]["name"])
        tgt = PG.slot_target(env, slots, k)
        for act in ex.plan(env.qpos(), bs.pos, bs.yaw, tgt, upper=slots[k]["layer"] > 0, far=slots[k]["row"] == 1):
            env.step(act)
        q0 = env.qpos()
        for i in range(30):
            env.step(q0 + (home - q0) * (i + 1) / 30)
        env.settle(15)
        ok.append(PG.evaluate(env, slots, k, refs, tgt)["success"])
    print("pallet slots ok:", ok)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.20 + 0.04, 0.03 + 0.02, 0.06]
    cam.distance, cam.azimuth, cam.elevation = 0.75, 205, -35
    big.update_scene(env.d, cam); save(big.render(), "pallet.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["basic", "object", "clutter", "factory", "looks", "pallet"])
    ap.add_argument("--name", default="obj")
    a = ap.parse_args()
    os.chdir(Path(__file__).resolve().parents[2])
    {"basic": basic, "object": lambda: obj(a.name), "clutter": clutter, "factory": factory_scene,
     "looks": looks, "pallet": pallet}[a.what]()
