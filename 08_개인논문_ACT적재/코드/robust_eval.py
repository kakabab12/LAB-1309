"""Robustness of trained stage policies to changes they never saw in the demonstrations.

Per-stage protocol of eval_act.py (same seeds, same success test), with one change applied per condition:
  base        no change (reference)
  pos_out     bin on the scale 22-28 mm off centre on one axis (demos: +-20 mm)
  yaw_out     bin rotated 22-30 deg (demos: +-20 deg)
  placed_out  earlier bins 10-14 mm off their targets (demos/eval: +-8 mm)            [stages 2-3]
  dark        all lights at 50 %
  bright      all lights at 160 %
  light_side  key light from the side instead of from above
  table_gray  table colour grey instead of light wood
  table_dark  table colour dark brown
  cam_top     fixed top camera moved 10 mm and tilted 2 deg
  cam_wrist   wrist camera moved 3 mm and tilted 2 deg
  distractor  three non-blue objects (red box, green cylinder, yellow ball) around the work area
  mass_100g   every bin weighs 100 g instead of 123 g (bins differ in weight in the real line)
  mass_200g   every bin weighs 200 g
  delay_67ms  the policy sees camera images and joint state 2 control steps (67 ms) late
  delay_133ms 4 control steps (133 ms) late
Beyond the domain-randomisation ranges of the v6 demonstrations (to show where robustness ends):
  very_dark   lights at 30 %        very_bright  lights at 220 %
  warm_light  orange-tinted lights  table_checker  grey checkerboard table
The policy runs on the CPU (3 chunk predictions per run), so this does not compete with GPU training.
usage: python robust_eval.py --ckpt S1 S2 S3 --tag v4_n100 --conds base dark ... --trials 20
"""

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch

from eval_act import EVAL_SEED0, Recorder, load_policy, run_stage
from expert import ScriptedExpert
from stack_env import BIN_NAMES, SCALE_C, TARGETS, StackEnv, sample_scene

STAGES = {"placed_out": (2, 3)}
MASS = {"mass_100g": 0.100, "mass_200g": 0.200}
DELAY = {"delay_67ms": 2, "delay_133ms": 4}
DISTRACT = [((0.33, -0.12, 0.025), (0.85, 0.15, 0.10)), ((0.34, 0.12, 0.025), (0.15, 0.70, 0.20)),
            ((0.20, 0.28, 0.025), (0.95, 0.85, 0.10))]  # (position, colour) for distract0..2


def quat_tilt(q: np.ndarray, deg: float, axis) -> np.ndarray:
    a = np.radians(deg) / 2
    r = np.array([math.cos(a), *(math.sin(a) * np.asarray(axis, float) / np.linalg.norm(axis))])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, q, r)
    return out


class Perturb:
    """Applies one visual/physical change to the model and undoes it afterwards."""

    def __init__(self, env: StackEnv):
        self.env, m = env, env.m
        self.saved = {k: getattr(m, k).copy() for k in ("light_diffuse", "light_dir", "light_ambient", "mat_rgba",
                                                         "mat_texid", "mat_texrepeat", "cam_pos", "cam_quat",
                                                         "body_mass", "body_inertia")}
        self.head = (m.vis.headlight.diffuse.copy(), m.vis.headlight.ambient.copy())
        self.tex_id = m.texture("tabletex").id
        self.tex0 = m.tex_data.copy()
        self.renderers = []  # renderers whose GPU copy of the table texture must follow tex_data

    def _upload_texture(self) -> None:
        for r in self.renderers:
            r._gl_context.make_current()
            mujoco.mjr_uploadTexture(self.env.m, r._mjr_context, self.tex_id)

    def apply(self, cond: str) -> None:
        m = self.env.m
        tab = m.material("table").id
        if cond in ("dark", "bright", "very_dark", "very_bright"):
            f = {"dark": 0.5, "bright": 1.6, "very_dark": 0.3, "very_bright": 2.2}[cond]
            m.light_diffuse[:] = self.saved["light_diffuse"] * f
            m.vis.headlight.diffuse[:] = self.head[0] * f
            m.vis.headlight.ambient[:] = self.head[1] * f
        elif cond == "warm_light":
            tint = np.array([1.25, 0.8, 0.45])
            m.light_diffuse[:] = self.saved["light_diffuse"] * tint
            m.vis.headlight.diffuse[:] = self.head[0] * tint
            m.vis.headlight.ambient[:] = self.head[1] * tint
        elif cond == "table_checker":
            w, h = m.tex_width[self.tex_id], m.tex_height[self.tex_id]
            yy, xx = np.mgrid[0:h, 0:w]
            cell = ((xx // 4) + (yy // 4)) % 2  # 16 x 16 squares per texture tile
            img = np.where(cell[..., None] == 0, [190, 190, 186], [95, 95, 92]).astype(np.uint8)
            a = m.tex_adr[self.tex_id]
            m.tex_data[a:a + img.size] = img.reshape(-1)
            self._upload_texture()
        elif cond == "light_side":
            d = np.array([0.45, 0.30, -0.84])
            m.light_dir[0] = d / np.linalg.norm(d)
        elif cond in ("table_gray", "table_dark"):
            m.mat_texid[tab, :] = -1
            m.mat_rgba[tab] = (0.55, 0.56, 0.58, 1) if cond == "table_gray" else (0.30, 0.20, 0.13, 1)
        elif cond == "cam_top":
            c = m.camera("top").id
            m.cam_pos[c] = self.saved["cam_pos"][c] + np.array([0.010, 0.0, 0.0])
            m.cam_quat[c] = quat_tilt(self.saved["cam_quat"][c], 2.0, (1, 0, 0))
        elif cond in MASS:
            for n in BIN_NAMES:
                b = m.body(n).id
                f = MASS[cond] / self.saved["body_mass"][b]
                m.body_mass[b] = MASS[cond]
                m.body_inertia[b] = self.saved["body_inertia"][b] * f
            mujoco.mj_setConst(m, self.env.d)
        elif cond == "cam_wrist":
            c = m.camera("front").id
            m.cam_pos[c] = self.saved["cam_pos"][c] + np.array([0.003, 0.0, 0.0])
            m.cam_quat[c] = quat_tilt(self.saved["cam_quat"][c], 2.0, (1, 0, 0))

    def reset(self) -> None:
        m = self.env.m
        for k, v in self.saved.items():
            getattr(m, k)[:] = v
        m.vis.headlight.diffuse[:], m.vis.headlight.ambient[:] = self.head
        mujoco.mj_setConst(m, self.env.d)
        if not np.array_equal(m.tex_data, self.tex0):
            m.tex_data[:] = self.tex0
            self._upload_texture()


def scene(cond: str, stage: int, i: int):
    rng = np.random.default_rng(EVAL_SEED0 + 1000 * stage + i)
    spec = sample_scene(stage, rng)
    alt = np.random.default_rng(900000 + 1000 * stage + i)  # separate stream for the out-of-range draws
    if cond == "pos_out":
        off = alt.uniform(-0.028, 0.028, size=2)
        k = alt.integers(2)
        off[k] = alt.choice([-1, 1]) * alt.uniform(0.022, 0.028)
        spec.new_bin_xy = SCALE_C + off
    elif cond == "yaw_out":
        spec.new_bin_yaw = float(alt.choice([-1, 1]) * np.radians(alt.uniform(22, 30)))
    elif cond == "placed_out":  # bin A (the base of the stack) 10-14 mm off its target
        p, yaw = spec.placed["bin_a"]
        ang = alt.uniform(0, 2 * np.pi)
        d = alt.uniform(0.010, 0.014) * np.array([np.cos(ang), np.sin(ang)])
        if stage == 2 and d[1] > 0.004:  # never push A into B's (fixed) target slot
            d[1] = -d[1]
        q = TARGETS[1].copy()
        q[:2] += d
        spec.placed["bin_a"] = (q, yaw)
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs=3, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--conds", nargs="+", required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--gifs", type=int, default=1, help="GIFs of the first k trials per condition and stage")
    a = ap.parse_args()

    torch.set_num_threads(2)
    out_dir = Path("results/robust") / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    env = StackEnv(render=True, dr=True)  # dr model only adds the (parked, invisible) distractor bodies
    home = ScriptedExpert(env).home_q()
    pols = [load_policy(p, "cpu", None) for p in a.ckpt]
    pert = Perturb(env)
    rec = Recorder(env, enabled=a.gifs > 0)
    pert.renderers = [env.renderer] + ([rec.big, rec.small] if a.gifs > 0 else [])
    for cond in a.conds:
        f = out_dir / f"{cond}.json"
        if f.exists():
            continue
        pert.reset()
        pert.apply(cond)
        res = {}
        for stage in STAGES.get(cond, (1, 2, 3)):
            rows = []
            for i in range(a.trials):
                spec = scene(cond, stage, i)
                env.reset(spec, home)
                if cond == "distractor":
                    for k, (pos, rgb) in enumerate(DISTRACT):
                        env.d.mocap_pos[env.m.body_mocapid[env.m.body(f"distract{k}").id]] = pos
                        env.m.geom_rgba[env.m.geom(f"distract{k}_g").id] = (*rgb, 1)
                    mujoco.mj_forward(env.m, env.d)
                refs = {n: env.bin_state(n).pos.copy() for n in BIN_NAMES[: stage - 1]}
                rec.enabled = i < a.gifs
                info = run_stage(env, pols[stage - 1], home, "cpu", rec, obs_delay=DELAY.get(cond, 0))
                r = env.evaluate_stage(stage, ref_positions=refs)
                rec.save(out_dir / "gifs" / f"{cond}_s{stage}_t{i:02d}_{'ok' if r['success'] else 'fail'}.gif")
                r.update(info, trial=i)
                rows.append(r)
            res[stage] = {"success_rate": float(np.mean([r["success"] for r in rows])), "n": len(rows), "trials": rows}
            print(f"{a.tag} {cond} stage {stage}: {100 * res[stage]['success_rate']:.0f}% "
                  f"({sum(r['success'] for r in rows)}/{len(rows)})", flush=True)
        f.write_text(json.dumps({"cond": cond, "ckpt": a.ckpt, "per_stage": res}, default=float))
    pert.reset()
    rec.close()
    env.close()


if __name__ == "__main__":
    main()
