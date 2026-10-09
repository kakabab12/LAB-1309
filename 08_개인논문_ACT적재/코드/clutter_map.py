"""Workspace map for the clutter test: how low the arm and the carried bins come over each spot of the table.

The scripted expert runs every stage on many scenes (the wide domain-randomisation pick range, so the map covers
more than the evaluation scenes), followed by the same return-to-home move as the policy runner. At every control
step the world-frame bounding box of each robot geom and each bin geom is drawn into a 5 mm grid that keeps the
lowest height seen per cell. The scale and the pallet plate are drawn as height 0 (nothing may stand there).
A clutter object of height h can stand at a spot if, within its footprint plus a margin, the map stays above
h + 15 mm (StackEnv.place_clutter). The same map tells a real cell where objects must not be put down.

usage: [ACT_OBJECT=cup ACT_LAYOUT=mirror ACT_DESIGN=cuprule] python clutter_map.py [--episodes 40]
       -> results/clutter_keepout[_cup_mirror].npz / .png
"""

import argparse
import json
import os

import mujoco
import numpy as np

from expert import ScriptedExpert, stage2_target
from stack_env import BIN_H, BIN_NAMES, KEEPOUT_FILE, ROOT, SCALE_C, SCALE_HALF, TARGETS, StackEnv, sample_scene

X0, Y0, RES, NX, NY = -0.15, -0.45, 0.005, 140, 180
SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)


def draw(env: StackEnv, geoms: np.ndarray, zmin: np.ndarray, zmax: np.ndarray | None = None) -> None:
    """zmin: lowest point seen per cell; zmax: highest (for structures above the cell, e.g. cables)."""
    m, d = env.m, env.d
    for g in geoms:
        c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
        w = d.geom_xpos[g] + (c + SIGNS * h) @ d.geom_xmat[g].reshape(3, 3).T
        i0, i1 = int((w[:, 0].min() - X0) // RES), int((w[:, 0].max() - X0) // RES)
        j0, j1 = int((w[:, 1].min() - Y0) // RES), int((w[:, 1].max() - Y0) // RES)
        i0, j0, i1, j1 = max(i0, 0), max(j0, 0), min(i1, NX - 1), min(j1, NY - 1)
        if i0 > i1 or j0 > j1:
            continue
        zmin[i0:i1 + 1, j0:j1 + 1] = np.minimum(zmin[i0:i1 + 1, j0:j1 + 1], max(w[:, 2].min(), 0.0))
        if zmax is not None:
            zmax[i0:i1 + 1, j0:j1 + 1] = np.maximum(zmax[i0:i1 + 1, j0:j1 + 1], w[:, 2].max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40, help="expert runs per stage")
    ap.add_argument("--seed", type=int, default=880000)
    a = ap.parse_args()
    env = StackEnv(render=False)
    m = env.m
    root = m.body_rootid[m.jnt_bodyid[m.joint("shoulder_pan").id]]
    robot = np.array([g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == root])
    bins = np.array([g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in BIN_NAMES])
    static = np.array([m.geom("scale_body").id, m.geom("pallet").id])
    zmin = np.full((NX, NY), np.inf)
    zmax = np.full((NX, NY), -np.inf)
    rng = np.random.default_rng(a.seed)
    ex = ScriptedExpert(env, rng)
    home = ex.home_q()
    stats = {}
    for stage in (1, 2, 3):
        ok = 0
        for _ in range(a.episodes):
            spec = sample_scene(stage, rng, wide=True)
            env.reset(spec, home)
            mujoco.mj_forward(m, env.d)
            draw(env, static, zmin)
            bs = env.bin_state(BIN_NAMES[stage - 1])
            a_now = env.bin_state("bin_a").pos
            tgt = {1: TARGETS[1], 2: stage2_target(a_now), 3: a_now + np.array([0, 0, BIN_H])}[stage]
            for act in ex.plan(env.qpos(), bs.pos, bs.yaw, tgt):
                env.step(act)
                draw(env, robot, zmin, zmax)
                draw(env, bins, zmin, zmax)
            q0 = env.qpos()
            for i in range(30):  # RETURN_HOME, as in eval_act.run_stage
                env.step(q0 + (home - q0) * (i + 1) / 30)
                draw(env, robot, zmin, zmax)
            env.settle(10)
            ok += env.evaluate_stage(stage)["success"]
        stats[stage] = ok / a.episodes
        print(f"stage {stage}: expert success {ok}/{a.episodes}", flush=True)
    KEEPOUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = KEEPOUT_FILE.with_name(KEEPOUT_FILE.name + ".tmp")  # running jobs may be reading the map
    with open(tmp, "wb") as f:
        np.savez(f, x0=X0, y0=Y0, res=RES, zmin=zmin, zmax=zmax)
    os.replace(tmp, KEEPOUT_FILE)
    free = np.isinf(zmin)
    print(json.dumps({"expert_success": stats, "cells_never_reached": float(free.mean())}))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    show = np.where(free, np.nan, np.clip(zmin, 0, 0.20) * 100).T
    im = ax.imshow(show, origin="lower", extent=(X0, X0 + NX * RES, Y0, Y0 + NY * RES), cmap="magma", vmin=0, vmax=20)
    fig.colorbar(im, ax=ax, label="lowest arm / carried-bin height (cm)")
    ax.add_patch(Rectangle((0.0, -0.34), 0.42, 0.68, fill=False, ls="--", ec="tab:blue", lw=1.2))
    ax.text(0.005, 0.345, "clutter area (top camera)", color="tab:blue", fontsize=8, va="bottom")
    ax.add_patch(Rectangle(SCALE_C - SCALE_HALF, *(2 * SCALE_HALF), fill=False, ec="w", lw=1))
    for s, t in TARGETS.items():
        if s < 3:
            ax.plot(*t[:2], "c+", ms=10)
    ax.set_xlabel("x (m, away from the robot)")
    ax.set_ylabel("y (m)")
    ax.set_title("Where the arm goes (expert runs, all stages)\nwhite = never reached")
    fig.tight_layout()
    fig.savefig(KEEPOUT_FILE.with_suffix(".png"), dpi=150)


if __name__ == "__main__":
    main()
