#!/usr/bin/env python
"""여러 모양·N개 장면 4K 시연 (10/24): 시범 프로그램(또는 VLA)으로 일 k 개를 끼어들기 겹쳐 하는 모습을 3840×2160 으로.
   로봇 둘레 전체가 보이게 위에서 비스듬히, 오른쪽 위 손목 카메라, 아래에 지금 따르는 지시(한국어)와 단계.
   PIPER_NOPED=1 MUJOCO_GL=egl python piper_sim/world_demo4k.py --seed 3 --k 3 --n-objects 8 --name world_expert_k3"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import piper_sim.piper_robot as pr  # noqa: E402

pr.use_piper_in_lerobot()
import piper_sim.world as W  # noqa: E402

W.install()
import piper_sim.world_expert as WE  # noqa: E402
import render4k as R4  # noqa: E402
import scripted_expert as se  # noqa: E402
import world_cmd as WC  # noqa: E402

CAM = dict(azimuth=180.0, elevation=-58.0, distance=1.32, lookat=(0.0, 0.0, 0.93))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--n-objects", type=int, default=8)
    p.add_argument("--n-pads", type=int, default=None)
    p.add_argument("--every", type=int, default=4, help="몇 스텝마다 한 장 (10장/초 → 4 면 2배속)")
    p.add_argument("--name", default="world_demo")
    a = p.parse_args()
    import imageio
    se._rng = np.random.default_rng(0)
    sc = W.sample_scene(a.seed, n_objects=a.n_objects, n_pads=a.n_pads or max(2, a.k))
    tasks = W.chain(sc, a.k, np.random.default_rng(a.seed))
    ko = {t[2]: f"{WC.ko(t[0])}{WC.kc.josa(WC.ko(t[0]), '을', '를')} {WC.ko(t[1])}{'' if W.parse(t[1])[0] == 'pad' else ' 위'}에" for t in tasks}
    ep = WE.make_episode(sc, a.seed, tasks)
    out = Path("outputs/media") / f"{a.name}.mp4"
    wr = imageio.get_writer(out, fps=10, codec="libx264", quality=8, macro_block_size=8)
    st = {"lines": [], "n": 0, "depth": 0, "stack": []}
    title = f"시범 프로그램 · 물체 {len(sc.objects)}개 · 판 {len(sc.pads)}개 · 일 {len(tasks)}개 끼어들기"

    def say(lang):
        if st["stack"] and lang == st["stack"][-1]:
            return
        if lang in st["stack"]:            # 돌아오기
            while st["stack"] and st["stack"][-1] != lang:
                st["stack"].pop()
            st["lines"].append((f"이어서: {ko[lang]} 놓기", (46, 111, 191)))
        else:
            first = not st["stack"]
            st["stack"].append(lang)
            st["lines"].append(((f"{ko[lang]} 놓아줘" if first else f"잠깐, {ko[lang]} 먼저 놓아줘"), (11, 11, 11)))
    orig = ep.step

    def step(action, phase):
        orig(action, phase)
        st["n"] += 1
        if st["n"] % a.every == 0:
            depth = len(st["stack"])
            wr.append_data(R4.frame(ep, st["lines"], f"{st['n'] / 20:.1f}초 · 하고 있는 일 {depth}겹", title, CAM))
    ep.step = step
    ok, each = WE.run_chain(ep, tasks, nested=True, say=say)
    st["lines"].append((("모두 끝" if ok else "끝 (일부 실패)") + f": {sum(each)}/{len(each)}", (24, 120, 60) if ok else (191, 46, 46)))
    for _ in range(10):
        wr.append_data(R4.frame(ep, st["lines"], f"{st['n'] / 20:.1f}초", title, CAM))
    wr.close()
    mb, n, ev = R4.save_gif(str(out), str(out.with_suffix(".gif")))
    print(f"{'성공' if ok else '실패'} {each}, 영상 {out}, GIF {mb:.1f}MB {n}장 (every {ev})", flush=True)


if __name__ == "__main__":
    main()
