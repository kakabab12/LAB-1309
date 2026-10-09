"""Grid palletising (rows x cols x layers of bins) with the final demonstration design.

Reuses stack_env.StackEnv with its module settings re-bound for N bins and its own scene files, so the
3-bin scene used by the main experiments is never touched. Expert = final design: one fixed grasp wall,
9 mm clearance, approach every target from the robot side, and on upper layers back off 5 mm from the
wall before lifting. Placement order avoids carrying a bin over or reaching past bins already placed:
columns from the far side of the scale (+y) towards the scale, and in each column the far row first
(the fixed jaw sits on the robot side of the wall).

usage: python pallet_grid.py --rows 2 --cols 3 --layers 2 --x0 0.16 --y0 0.07 --seqs 3 [--gif]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("ACT_DESIGN", "v4")

import mujoco
import numpy as np

import expert as E
import stack_env as SE

PITCH = SE.BIN_W + 0.008  # row pitch (x)
# column pitch (y): gap between neighbouring columns, PALLET_GAP_Y (m) to widen it (default 8 mm like the rows)
PITCH_Y = SE.BIN_W + float(os.environ.get("PALLET_GAP_Y", "0.008"))


def configure(rows: int, cols: int, layers: int, x0: float, y0: float) -> list[dict]:
    """Re-bind stack_env for rows*cols*layers bins; returns the slots in placement order."""
    n = rows * cols * layers
    SE.BIN_NAMES = [f"bin_{i}" for i in range(n)]
    SE.PARK = [np.array([-0.6 - 0.12 * (i // 6), -0.3 + 0.12 * (i % 6), SE.BIN_H / 2]) for i in range(n)]
    tag = f"{rows}x{cols}x{layers}"
    SE.SCENE_OUT = SE.ROBOT_DIR / f"_pallet_{tag}_scene.xml"
    SE.SCENE_DR_OUT = SE.ROBOT_DIR / f"_pallet_{tag}_scene_dr.xml"
    slots = []
    for L in range(layers):
        for c in reversed(range(cols)):  # far side of the scale (+y) first
            for r in reversed(range(rows)):  # far row (+x) first
                slots.append({"layer": L, "row": r, "col": c,
                              "xy": np.array([x0 + r * PITCH, y0 + c * PITCH_Y])})
    for k, s in enumerate(slots):
        s["name"] = SE.BIN_NAMES[k]
        s["below"] = None if s["layer"] == 0 else next(
            j for j, t in enumerate(slots) if t["layer"] == s["layer"] - 1 and t["row"] == s["row"] and t["col"] == s["col"])
    return slots


class GridExpert(E.ScriptedExpert):
    EPISODE_STEPS = 300

    VIA_FAR_ONLY = os.environ.get("PALLET_VIA", "all") == "far"  # skip the robot-side via point for the near row

    def plan(self, q_start, bin_pos, bin_yaw, target, jitter=True, upper=False, far=True):
        rng = self.rng
        sp = (lambda: rng.uniform(0.9, 1.12)) if jitter else (lambda: 1.0)
        jx = (lambda s: rng.uniform(-s, s, size=3) * np.array([1, 1, 0.5])) if jitter else (lambda s: np.zeros(3))
        n_pick = self.facing_wall_normal(bin_pos[:2], bin_yaw)
        yaw_pick = math.atan2(n_pick[1], n_pick[0])
        yaw_place = min([math.pi, -math.pi], key=lambda c: abs(E.wrap(c - yaw_pick)))
        yaw_place = yaw_pick + E.wrap(yaw_place - yaw_pick)
        n_place = np.array([-1.0, 0.0, 0.0])
        off = SE.BIN_W / 2 + self.CLEAR
        rim_pick = bin_pos[2] + SE.BIN_H / 2
        rim_place = target[2] + SE.BIN_H / 2
        p0, _ = self.ik.fk(q_start)
        grasp = np.array([*(bin_pos[:2] + off * n_pick[:2]), rim_pick - self.GRASP_DEPTH]) + jx(0.0015)
        pre = grasp + np.array([0, 0, self.APPROACH_DZ]) + jx(0.006)
        lift = np.array([grasp[0], grasp[1], self.CARRY_Z]) + jx(0.006)
        place = np.array([*(target[:2] + (SE.BIN_W / 2) * n_place[:2]), rim_place - self.GRASP_DEPTH + 0.004]) + jx(0.0015)
        above = np.array([place[0], place[1], self.CARRY_Z]) + jx(0.006)
        stage_pt = np.array([place[0] - 0.05, place[1], self.CARRY_Z]) + jx(0.004)  # come in from the robot side
        backoff = place + (0.005 if upper else 0.0) * n_place
        retreat = backoff + np.array([0, 0, 0.06]) + jx(0.006)
        G_O, G_C = SE.GRIPPER_OPEN, SE.GRIPPER_CLOSED
        via = [(stage_pt, yaw_place, G_C, 1.6 * sp()), (above, yaw_place, G_C, 0.8 * sp())]
        if self.VIA_FAR_ONLY and not far:
            via = [(above, yaw_place, G_C, 2.4 * sp())]
        wps = [(pre, yaw_pick, G_O, 1.6 * sp()), (grasp, yaw_pick, G_O, 0.9 * sp()), (grasp, yaw_pick, G_C, 0.6 * sp()),
               (lift, yaw_pick, G_C, 0.9 * sp())] + via + [(place, yaw_place, G_C, 1.0 * sp()),
                                                           (place, yaw_place, G_O, 0.5 * sp())]
        wps += ([(backoff, yaw_place, G_O, 0.3 * sp()), (retreat, yaw_place, G_O, 0.6 * sp())] if upper
                else [(retreat, yaw_place, G_O, 0.8 * sp())])
        traj, q = [], q_start.copy()
        prev_p, prev_yaw, prev_g = p0, yaw_pick, q_start[5]
        max_err = 0.0
        for p, yaw, g, dur in wps:
            n = max(2, int(round(dur * SE.CONTROL_HZ)))
            for i in range(1, n + 1):
                s = E.min_jerk(i / n)
                pi_ = prev_p + s * (p - prev_p)
                yi = prev_yaw + s * (yaw - prev_yaw)
                gi = prev_g + min(1.0, i / max(1, n * 0.7)) * (g - prev_g)
                q, err = self.ik.solve(q, pi_, self.geo.R_target(yi, pi_[:2], pi_[2]))
                max_err = max(max_err, err)
                qq = q.copy()
                qq[5] = gi
                traj.append(qq)
            prev_p, prev_yaw, prev_g = p, yaw, g
        self.last_ik_err = max_err
        traj = traj[: self.EPISODE_STEPS]
        while len(traj) < self.EPISODE_STEPS:
            traj.append(traj[-1].copy())
        return np.array(traj, dtype=np.float32)


def slot_target(env, slots, k) -> np.ndarray:
    s = slots[k]
    if s["below"] is None:
        return np.array([*s["xy"], SE.BIN_H / 2])
    b = env.bin_state(slots[s["below"]]["name"]).pos
    return np.array([b[0], b[1], b[2] + SE.BIN_H])


def evaluate(env, slots, k, refs, target) -> dict:
    bs = env.bin_state(slots[k]["name"])
    xy = float(np.linalg.norm(bs.pos[:2] - target[:2]))
    z = float(abs(bs.pos[2] - target[2]))
    disturbed = [n for n, p in refs.items() if np.linalg.norm(env.bin_state(n).pos - p) > 0.010]
    ok = xy < 0.015 and z < 0.008 and bs.tilt_deg < 10.0 and not disturbed
    return {"success": bool(ok), "xy_err_mm": xy * 1000, "z_err_mm": z * 1000, "tilt_deg": bs.tilt_deg,
            "disturbed": disturbed}


_build_orig = SE.build_model_files


def _build_with_arena(dr: bool = False, clutter: bool = False):
    """Same scene, with a larger constraint arena (12+ bins in contact need more than the 16 MB default)."""
    path = _build_orig(dr, clutter)
    txt = path.read_text()
    if "<size memory" not in txt:
        txt = txt.replace("<option ", '<size memory="64M"/>\n  <option ', 1)
        SE._atomic_write(path, txt)
    return path


def make_env(slots, render=False):
    SE.build_model_files = _build_with_arena  # only in this process; the pallet scene files are pallet-only
    env = SE.StackEnv(render=render)
    m = env.m
    xs = [s["xy"][0] for s in slots]
    ys = [s["xy"][1] for s in slots]
    g = m.geom("pallet").id  # visual pallet under the whole grid
    m.geom_pos[g][:2] = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2]
    m.geom_size[g][:2] = [(max(xs) - min(xs)) / 2 + 0.04, (max(ys) - min(ys)) / 2 + 0.04]
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--x0", type=float, default=0.16)
    ap.add_argument("--y0", type=float, default=0.07)
    ap.add_argument("--seqs", type=int, default=3)
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=2024)
    a = ap.parse_args()
    slots = configure(a.rows, a.cols, a.layers, a.x0, a.y0)
    env = make_env(slots, render=False)
    ex = GridExpert(env, np.random.default_rng(11))
    home = ex.home_q()
    rng = np.random.default_rng(a.seed)
    tag = a.tag or (f"{a.rows}x{a.cols}x{a.layers}_x{a.x0:.3f}_y{a.y0:.3f}"
                    + ("_viafar" if GridExpert.VIA_FAR_ONLY else ""))
    frames, res = [], []
    if a.gif:
        from PIL import Image
        big = mujoco.Renderer(env.m, 360, 480)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [a.x0 + 0.04, a.y0 + 0.02, 0.06]
        cam.distance, cam.azimuth, cam.elevation = 0.75, 205, -35
    for seq in range(a.seqs):
        xy, yaw = SE.sample_pick(rng)
        env.reset(SE.SceneSpec(stage=1, new_bin_xy=xy, new_bin_yaw=yaw), home)
        row = []
        for k in range(len(slots)):
            if k > 0:
                xy, yaw = SE.sample_pick(rng)
                env.spawn_next_bin(k + 1, xy, yaw)
            refs = {s["name"]: env.bin_state(s["name"]).pos.copy() for s in slots[:k]}
            bs = env.bin_state(slots[k]["name"])
            tgt = slot_target(env, slots, k)
            for t, act in enumerate(ex.plan(env.qpos(), bs.pos, bs.yaw, tgt, upper=slots[k]["layer"] > 0,
                                            far=slots[k]["row"] == a.rows - 1)):
                env.step(act)
                if a.gif and seq == 0 and t % 6 == 0:
                    big.update_scene(env.d, cam)
                    frames.append(Image.fromarray(big.render()))
            q0 = env.qpos()
            for i in range(30):
                env.step(q0 + (home - q0) * (i + 1) / 30)
            env.settle(15)
            r = evaluate(env, slots, k, refs, tgt)
            r.update(slot=k, layer=slots[k]["layer"], row=slots[k]["row"], col=slots[k]["col"],
                     ik_err_mm=ex.last_ik_err * 1000)
            row.append(r)
        res.append(row)
        print(f"{tag} seq {seq}: " + " ".join(
            f"{x['slot']}:{'o' if x['success'] else 'x'}" + ("" if x["success"] else f"({x['xy_err_mm']:.0f}mm,{x['tilt_deg']:.0f}°,{','.join(x['disturbed'])})")
            for x in row), flush=True)
        if seq == 0 and a.gif:
            big.update_scene(env.d, cam)
            Image.fromarray(big.render()).save(f"results/pallet_{tag}_final.png")
    ok = np.array([[x["success"] for x in r] for r in res])
    ik = np.array([[x["ik_err_mm"] for x in r] for r in res])
    out = {"tag": tag, "rows": a.rows, "cols": a.cols, "layers": a.layers, "x0": a.x0, "y0": a.y0,
           "slot_success": ok.mean(0).tolist(), "full_stack": float(ok.all(1).mean()),
           "slot_ik_err_max_mm": ik.max(0).tolist(), "slots": [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                                              for k, v in s.items()} for s in slots], "runs": res}
    Path("results/pallet").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(f"results/pallet/expert_{tag}.json", "w"), indent=1, default=float)
    print(f"{tag}: per-slot {np.round(ok.mean(0), 2).tolist()} | full stack {ok.all(1).mean():.2f} | "
          f"max IK err {ik.max():.1f} mm", flush=True)
    if frames:
        frames[0].save(f"results/pallet_{tag}_expert.gif", save_all=True, append_images=frames[1:], duration=120,
                       loop=0, optimize=True)
    env.close()


if __name__ == "__main__":
    main()
