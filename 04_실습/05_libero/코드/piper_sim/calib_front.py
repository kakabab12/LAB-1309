#!/usr/bin/env python
"""정면 카메라 화면 위치 → 책상 위 블록 위치 변환식 맞추기 (10/15, 손목 정렬 보조 grasp_assist 용)
   블록마다 x,y = A @ [u, v, 1, u², uv, v²] (2차 — 직선만이면 책상 가장자리에서 최대 0.57cm, 원근 때문). 실물에서는 블록을 책상 몇 군데에 두고 같은 식으로 한 번 맞춘다.
   python piper_sim/calib_front.py --out piper_sim/front_calib.json"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import piper_sim.piper_robot as pr  # noqa: E402

pr.use_piper_in_lerobot()
import objdet  # noqa: E402
import sim_only as so  # noqa: E402
import switch_experiment as sx  # noqa: E402

NAMES = {"red": "akita_black_bowl_1", "green": "cream_cheese_1", "blue": "blue_block_1", "yellow": "wine_bottle_1"}


def feats(u, v):
    return [u, v, 1.0, u * u, u * v, v * v]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", default="5000-5039")
    p.add_argument("--out", default="piper_sim/front_calib.json")
    a = p.parse_args()
    lo, hi = map(int, a.episodes.split("-"))
    X = {c: [] for c in NAMES}
    Y = {c: [] for c in NAMES}
    for task in (0, 8):
        r = so.SimRunner(task)
        for i in range(lo, hi + 1):
            ep = sx.Episode(r, i)
            f = objdet.detect(np.asarray(ep.obs["pixels"]["image"]), np.asarray(ep.obs["pixels"]["image2"]))
            for k, c in enumerate(objdet.BLOCKS):
                if f[2 * k] < 0:
                    continue
                X[c].append(feats(f[2 * k], f[2 * k + 1]))
                Y[c].append(ep.obj_pos(NAMES[c])[:2])
            ep.env.close()
    out = {}
    for c in NAMES:
        A, *_ = np.linalg.lstsq(np.array(X[c]), np.array(Y[c]), rcond=None)
        res = np.linalg.norm(np.array(X[c]) @ A - np.array(Y[c]), axis=1)
        out[c] = A.T.tolist()
        print(f"{c}: {len(X[c])}장면, 남는 오차 평균 {100 * res.mean():.2f}cm 최대 {100 * res.max():.2f}cm", flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
