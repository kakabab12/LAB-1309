#!/usr/bin/env python
"""실패 장면 따라가기 (10/31): world_eval 과 같은 장면·일·동작 변수로 다시 돌리며 스텝마다 손 자리·VLA 동작·다듬기 출력·
   다듬기 상태(쥔 것, 놓을 곳 lock, 고른 물체)를 json 으로, 작은 영상(960×540, 손목 화면 붙임)을 mp4 로 남긴다.
   새 기준 w10m + 다듬기 일 1개에서 지시한 물체 3cm 안까지 가고도 600스텝 동안 안 쥔 장면이 105개 중 11개 — 왜 안 쥐는지 보려고.
   PIPER_NOPED=1 MUJOCO_GL=egl PIPER_WORLD=piper_sim/worlds/random.json python piper_sim/world_trace.py --policy outputs/piper_w10m_model/merged \
       --seeds 1004,1017 --k 1 --nested --name tr_w10m_random --refine grasp,place,... --latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import piper_sim.world_eval as WEv  # noqa: E402  (piper_robot·world 설치까지 한다)
import render4k as R4  # noqa: E402

W, sx = WEv.W, WEv.sx
CAM = dict(azimuth=180.0, elevation=-58.0, distance=1.32, lookat=(0.0, 0.0, 0.93))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--seeds", required=True, help="1004,1017 (world_eval 과 같은 꼴)")
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--nested", action="store_true")
    p.add_argument("--n-objects", type=int, default=None)
    p.add_argument("--n-pads", type=int, default=None)
    p.add_argument("--switch-hold", type=int, default=30)
    p.add_argument("--release-wait", type=int, default=40)
    p.add_argument("--no-release-refine", dest="release_refine", action="store_false")
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--every", type=int, default=2, help="몇 스텝마다 영상 한 장 (0 이면 영상 없이 기록만)")
    p.add_argument("--name", default="world_trace")
    p.add_argument("--refine", default=None)
    a, rest = p.parse_known_args()
    args = sx.fill_defaults(sx.build_parser().parse_args(["--task-a", "0", "--policy", a.policy] + rest))
    args.task_a, args.task_b, args.strategy = 0, None, "none"
    r = sx.Runner(args)
    WEv.pad_lang(r)
    if a.refine:
        import piper_sim.world_refine as WR
        r.assist = WR.make(tuple(a.refine.split(",")))
    for seed in WEv.rng_list(a.seeds):
        trace(a, r, seed)


def trace(a, r, seed):
    rng = np.random.default_rng(seed * 31 + a.k)                    # world_eval.main 과 같은 장면·일
    lo, hi = W.CFG["n_objects"]
    n = a.n_objects or int(rng.integers(max(lo, a.k), max(hi, a.k) + 1))
    sc = W.sample_scene(seed, n_objects=n, n_pads=a.n_pads or W.pick_n_pads(rng, a.k))
    tasks = W.chain(sc, a.k, rng)
    out = Path("outputs/media/trace") / f"{a.name}_{seed}"
    out.parent.mkdir(parents=True, exist_ok=True)
    wr = None
    if a.every:
        import imageio
        R4.W, R4.H = 960, 540
        wr = imageio.get_writer(str(out) + ".mp4", fps=10, codec="libx264", quality=7, macro_block_size=8)
    rows, cur = [], {}
    ra = r.assist
    if ra is not None:                                               # 다듬기 앞뒤 동작을 잡는다
        orig_call = type(ra).__call__

        def call(self, ep, act, instruction):
            o = orig_call(self, ep, act, instruction)
            cur.update(vla=[round(float(v), 3) for v in np.asarray(act)[[0, 1, 2, 6]]],
                       out=[round(float(v), 3) for v in np.asarray(o)[[0, 1, 2, 6]]],
                       held=getattr(self, "held", None), lock=getattr(self, "lock", None), want=getattr(self, "_k_now", None),
                       gplan=None if getattr(self, "gplan", None) is None or self.gplan.plan is None else "on",
                       pf=None if getattr(self, "_pf", None) is None else self._pf.get("phase"),
                       mem=None if tasks[0][0] not in getattr(self, "pos", {}) else [round(float(v), 4) for v in self.pos[tasks[0][0]]],
                       fresh=None if tasks[0][0] not in (getattr(self, "fresh", None) or {}) else [round(float(v), 4) for v in self.fresh[tasks[0][0]]],
                       gp=None if getattr(self, "gplan", None) is None or self.gplan.plan is None else
                       [self.gplan.plan["phase"]] + [round(float(v), 4) for v in self.gplan.plan["g"]])
            return o
        type(ra).__call__ = call
    orig_step = sx.Episode.step if not hasattr(sx.Episode, "_orig_step") else sx.Episode._orig_step
    sx.Episode._orig_step = orig_step

    def step(self, action, phase):
        orig_step(self, action, phase)
        t = len(rows)
        hand = np.asarray(self.obs["robot_state"]["eef"]["pos"], dtype=np.float64)
        sim, inner = self.inner.sim, self.inner
        o, d = tasks[0][0], tasks[0][1]
        po = sim.data.body_xpos[inner.obj_body_id[o]]
        pd = sim.data.body_xpos[inner.obj_body_id[d]]
        row = dict(t=t, ph=phase, hand=[round(float(v), 3) for v in hand], obj=[round(float(v), 3) for v in po],
                   d_obj=round(100 * float(np.linalg.norm(po[:2] - hand[:2])), 1), dz=round(100 * float(hand[2] - po[2]), 1),
                   d_dest=round(100 * float(np.linalg.norm(po[:2] - pd[:2])), 1), hold=bool(self.holding()), fo=self.finger_object(),
                   q=[round(float(v), 4) for v in self.obs["robot_state"]["gripper"]["qpos"]],
                   tilt=round(float(np.degrees(np.arccos(np.clip(sim.data.body_xmat[inner.obj_body_id[o]].reshape(3, 3)[2, 2], -1, 1)))), 1))
        row.update(cur)
        cur.clear()
        rows.append(row)
        if wr is not None and (t % a.every == 0 or phase == "E"):
            img, wrist = R4.render(self, CAM)
            from PIL import Image, ImageDraw
            im = Image.fromarray(img)
            im.paste(Image.fromarray(wrist).resize((240, 240)), (960 - 250, 10))
            dr = ImageDraw.Draw(im)
            txt = f"{t} {phase} 물체 {row['d_obj']}cm 높이차 {row['dz']} 놓을곳 {row['d_dest']}cm 쥠 {row['fo'] or '-'} " \
                  f"VLA g {row.get('vla', [0] * 4)[3]} → {row.get('out', [0] * 4)[3]} lock {row.get('lock') or '-'} pf {row.get('pf') or '-'}"
            dr.rectangle([0, 540 - 28, 960, 540], fill=(252, 252, 251))
            dr.text((8, 540 - 24), txt, fill=(11, 11, 11), font=_font())
            wr.append_data(np.asarray(im))

    sx.Episode.step = step
    ea = SimpleNamespace(nested=a.nested, switch_hold=a.switch_hold, max_steps=a.max_steps, release_wait=a.release_wait,
                         release_refine=a.release_refine)
    try:
        res = WEv.run_episode(r, sc, tasks, ea, seed)
    finally:
        sx.Episode.step = orig_step
        if ra is not None:
            type(ra).__call__ = orig_call
        if wr is not None:
            wr.close()
    Path(str(out) + ".json").write_text(json.dumps({"tasks": tasks, "result": res, "rows": rows}, ensure_ascii=False))
    print(f"{'성공' if res['all_ok'] else '실패'} {tasks} 스텝 {len(rows)} → {out}.json" + (f", {out}.mp4" if wr is not None else ""), flush=True)


_F = []


def _font():
    if not _F:
        from PIL import ImageFont
        _F.append(ImageFont.truetype(R4.FONT_R, 16))
    return _F[0]


if __name__ == "__main__":
    main()
