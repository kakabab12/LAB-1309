#!/usr/bin/env python
"""시범 프로그램이 여러 모양·N개 장면에서 일 k 개를 (끼어들기 겹쳐서 / 차례로) 해내는지 (10/24).
   PIPER_NOPED=1 MUJOCO_GL=egl python piper_sim/world_chain_test.py --seeds 2000-2009 --k 2 3 4 --nested --video outputs/media/world_chain.mp4"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import piper_sim.piper_robot as pr  # noqa: E402

pr.use_piper_in_lerobot()
import piper_sim.world as W  # noqa: E402

W.install()
import piper_sim.world_expert as WE  # noqa: E402
import scripted_expert as se  # noqa: E402
import switch_experiment as sx  # noqa: E402


def rng_list(s):
    out = []
    for part in s.split(","):
        a, b = part.split("-") if "-" in part else (part, part)
        out += list(range(int(a), int(b) + 1))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="2000-2004")
    p.add_argument("--k", type=int, nargs="+", default=[2, 3])
    p.add_argument("--nested", action="store_true", help="끼어들기 겹치기 (아니면 차례로)")
    p.add_argument("--n-objects", type=int, default=None, help="장면 물체 수 (없으면 max(k, 설정 범위))")
    p.add_argument("--video", default=None, help="첫 장면 mp4")
    p.add_argument("--out", default="outputs/piper/world_chain.json")
    a = p.parse_args()
    se._rng = np.random.default_rng(0)
    res = []
    for k in a.k:
        for i, seed in enumerate(rng_list(a.seeds)):
            rng = np.random.default_rng(seed * 31 + k)
            n = a.n_objects or max(k, int(rng.integers(*W.CFG["n_objects"]) + 1))
            sc = W.sample_scene(seed, n_objects=n, n_pads=min(len(W.CFG["pad_colors"]), max(2, k)))
            tasks = W.chain(sc, k, rng)
            args = sx.fill_defaults(sx.default_args())
            args.video = bool(a.video and i == 0 and k == a.k[-1])
            t0 = time.time()
            ep = WE.make_episode(sc, seed, tasks, args)
            log = []
            try:
                ok, each = WE.run_chain(ep, tasks, nested=a.nested, log=log)
            except Exception as e:
                ok, each = False, [False] * len(tasks)
                log.append(("예외", f"{type(e).__name__}: {e}"[:200]))
            dt = time.time() - t0
            r = {"k": k, "seed": seed, "n_objects": len(sc.objects), "n_pads": len(sc.pads), "ok": ok, "each": each,
                 "steps": len(ep.acts), "sec": round(dt, 1), "tasks": [t[2] for t in tasks], "log": log}
            res.append(r)
            print(f"k={k} 장면 {seed} (물체 {len(sc.objects)}, 판 {len(sc.pads)}): {'성공' if ok else '실패'} {each} "
                  f"{len(ep.acts)}스텝 {dt:.0f}초 | {log[-3:]}", flush=True)
            if args.video and ep.frames:
                import imageio
                Path(a.video).parent.mkdir(parents=True, exist_ok=True)
                imageio.mimsave(a.video, ep.frames[::2], fps=20)
                print("영상:", a.video, flush=True)
            ep.env.close()
    for k in a.k:
        rk = [r for r in res if r["k"] == k]
        print(f"== k={k}: 전부 성공 {sum(r['ok'] for r in rk)}/{len(rk)}, 일별 {sum(sum(r['each']) for r in rk)}/{sum(len(r['each']) for r in rk)}",
              flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
