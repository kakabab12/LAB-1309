#!/usr/bin/env python
"""여러 모양·N개 장면 VLA(+보조) 4K 시연 (10/25): world_eval 과 같은 장면·일(같은 seed·k)을 같은 평가 설정으로 다시 돌리며
   3840×2160 으로 찍는다 (world_demo4k 는 시범 프로그램). 평가 인자(--latency-steps … 보조 옵션)는 world_eval 과 똑같이 넘긴다.
   PIPER_NOPED=1 MUJOCO_GL=egl python piper_sim/world_vla4k.py --policy outputs/piper_w1_model/merged --seed 1003 --k 2 --nested \
       --name world_w1a_k2 --label "VLA + 보조" --latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph [보조 옵션]"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import piper_sim.world_eval as WEv  # noqa: E402  (piper_robot·world 설치까지 한다)
import render4k as R4  # noqa: E402
import world_cmd as WC  # noqa: E402

W, sx = WEv.W, WEv.sx
CAM = dict(azimuth=180.0, elevation=-58.0, distance=1.32, lookat=(0.0, 0.0, 0.93))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--nested", action="store_true")
    p.add_argument("--n-objects", type=int, default=None)
    p.add_argument("--switch-hold", type=int, default=30)
    p.add_argument("--release-wait", type=int, default=40)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--every", type=int, default=4, help="몇 스텝마다 한 장 (10장/초 → 4 면 2배속)")
    p.add_argument("--name", default="world_vla")
    p.add_argument("--label", default="VLA")
    a, rest = p.parse_known_args()
    import imageio
    args = sx.fill_defaults(sx.build_parser().parse_args(["--task-a", "0", "--policy", a.policy] + rest))
    args.task_a, args.task_b, args.strategy = 0, None, "none"
    r = sx.Runner(args)
    if r.assist is not None:
        import piper_sim.world_assist as WA
        r.assist = WA.from_args(args)
    # world_eval.main 과 같은 장면·일
    rng = np.random.default_rng(a.seed * 31 + a.k)
    lo, hi = W.CFG["n_objects"]
    n = a.n_objects or int(rng.integers(max(lo, a.k), max(hi, a.k) + 1))
    sc = W.sample_scene(a.seed, n_objects=n, n_pads=min(len(W.CFG["pad_colors"]), max(2, a.k)))
    tasks = W.chain(sc, a.k, rng)
    ko = {t[2]: f"{WC.ko(t[0])}{WC.kc.josa(WC.ko(t[0]), '을', '를')} {WC.ko(t[1])}{'' if W.parse(t[1])[0] == 'pad' else ' 위'}에" for t in tasks}
    out = Path("outputs/media") / f"{a.name}.mp4"
    wr = imageio.get_writer(out, fps=10, codec="libx264", quality=8, macro_block_size=8)
    st = {"lines": [], "n": 0, "stack": []}
    title = f"{a.label} · 물체 {len(sc.objects)}개 · 판 {len(sc.pads)}개 · 일 {len(tasks)}개" + (" 끼어들기" if a.nested and a.k > 1 else "")

    def say(lang):
        if lang not in ko or (st["stack"] and lang == st["stack"][-1]):
            return
        if lang in st["stack"]:            # 돌아오기
            while st["stack"] and st["stack"][-1] != lang:
                st["stack"].pop()
            st["lines"].append((f"이어서: {ko[lang]} 놓기", (46, 111, 191)))
        else:
            first = not st["stack"]
            st["stack"].append(lang)
            st["lines"].append(((f"{ko[lang]} 놓아줘" if first else f"잠깐, {ko[lang]} 먼저 놓아줘"), (11, 11, 11)))

    orig_step, orig_rp = sx.Episode.step, sx.Episode.run_policy

    def step(self, action, phase):
        orig_step(self, action, phase)
        st["n"] += 1
        if phase == "E" and not st.get("end"):
            st["end"] = True
            fin = [bool(WEv.WE.Goal(o, d)(self.env)) for o, d, _ in tasks]
            st["lines"].append(((("모두 끝" if all(fin) else "끝 (일부 실패)") + f": {sum(fin)}/{len(fin)}"),
                                (24, 120, 60) if all(fin) else (191, 46, 46)))
        if st["n"] % a.every == 0 or phase == "E":
            wr.append_data(R4.frame(self, st["lines"], f"{st['n'] / 20:.1f}초 · 하고 있는 일 {len(st['stack'])}겹", title, CAM))

    def run_policy(self, instruction, tag, *x, **kw):
        say(instruction)
        return orig_rp(self, instruction, tag, *x, **kw)

    sx.Episode.step, sx.Episode.run_policy = step, run_policy
    ea = SimpleNamespace(nested=a.nested, switch_hold=a.switch_hold, max_steps=a.max_steps, release_wait=a.release_wait)
    d = WEv.run_episode(r, sc, tasks, ea, a.seed)
    wr.close()
    mb, nf, ev = R4.save_gif(str(out), str(out.with_suffix(".gif")))
    print(f"{'성공' if d['all_ok'] else '실패'} final {d['final']} 판 안 {d['final_loose']}, 영상 {out}, GIF {mb:.1f}MB {nf}장 (every {ev})", flush=True)


if __name__ == "__main__":
    main()
