#!/usr/bin/env python
"""쌓은 블록용 정면 카메라 변환식 (10/23). 한국어 실행기(korean_live)가 '초록을 파랑 위에' 를 다 하고도 끝난 줄 몰랐다
   (장면 1004 4K 시연, 60초에 쌓았는데 120초까지 맴돎). 책상 전체로 맞춘 식(front_calib_wide.json)은 블록이 책상 위에 있다고 보고 맞춰,
   4.5cm 위에 얹힌 블록과 그 블록에 반쯤 가린 아래 블록을 어긋나게 읽는다.
   블록 하나를 아무 데나 놓고 다른 블록을 그 위에 얹어(±1cm, 방향 아무렇게나) 찍어, '위 블록' 식과 '가린 아래 블록' 식을 따로 맞춘다.
   python piper_sim/calib_front_stack.py --out piper_sim/front_calib_stack.json"""
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
from piper_sim.calib_front import NAMES, feats  # noqa: E402
from piper_sim.calib_front_wide import fit  # noqa: E402

CUBES = ["red", "green", "blue"]
PARK = {"red": (-0.15, 0.25), "green": (-0.15, -0.25), "blue": (0.05, 0.25), "yellow": (0.05, -0.27)}     # 쓰지 않는 블록은 구석으로


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=240)
    p.add_argument("--out", default="piper_sim/front_calib_stack.json")
    p.add_argument("--wide", default="piper_sim/front_calib_wide.json")
    a = p.parse_args()
    rng = np.random.default_rng(23)
    W = {k: np.array(v) for k, v in json.loads(Path(a.wide).read_text()).items()}
    Xt, Yt, Xb, Yb, D = [], [], [], [], []
    r = so.SimRunner(8)
    ep = sx.Episode(r, 6000)
    for i in range(a.n):
        sim = ep.inner.sim
        bot, top = rng.choice(CUBES, 2, replace=False)
        while True:
            q = np.array([rng.uniform(-0.18, 0.10), rng.uniform(-0.28, 0.28)])
            if 0.18 < np.linalg.norm(q - [-0.33, 0]) < 0.40:
                break
        place = {bot: (q, 0.93), top: (q + rng.uniform(-0.01, 0.01, 2), 0.93 + 0.046)}
        for c in PARK:                     # 블록 4개만 (판은 그대로)
            o = ep.inner.objects_dict[NAMES[c]]
            qp = sim.data.get_joint_qpos(o.joints[0]).copy()
            xy, z = place.get(c, (np.array(PARK[c]), 0.93))
            th = rng.uniform(-np.pi, np.pi)
            qp[:3] = [xy[0], xy[1], z]
            qp[3:] = [np.cos(th / 2), 0, 0, np.sin(th / 2)]
            sim.data.set_joint_qpos(o.joints[0], qp)
        sim.forward()
        for _ in range(15):
            obs, *_ = ep.env.step(np.zeros(7))
        tb, bb = ep.obj_pos(NAMES[top]), ep.obj_pos(NAMES[bot])
        if not (np.linalg.norm(tb[:2] - bb[:2]) < 0.02 and tb[2] - bb[2] > 0.035):     # 굴러떨어졌으면 뺌
            continue
        f = objdet.detect(np.asarray(obs["pixels"]["image"]), np.asarray(obs["pixels"]["image2"]))
        k_t, k_b = objdet.OBJECTS.index(top), objdet.OBJECTS.index(bot)
        if f[2 * k_t] < 0 or f[2 * k_b] < 0:
            continue
        ft, fb = feats(f[2 * k_t], f[2 * k_t + 1]), feats(f[2 * k_b], f[2 * k_b + 1])
        Xt.append(ft); Yt.append(tb[:2]); Xb.append(fb); Yb.append(bb[:2])
        D.append(np.linalg.norm(W[top] @ ft - W[bot] @ fb))      # 책상 식으로 잰 위·아래 거리 (끝 판정이 본 값)
        if i % 40 == 39:
            print(f"{i + 1}번, 쓸 수 있는 점 {len(Xt)}", flush=True)
    D = np.array(D)
    print(f"책상 식으로 잰 위·아래 블록 거리: 평균 {100 * D.mean():.2f}cm, 중앙 {100 * np.median(D):.2f}cm, 최대 {100 * D.max():.2f}cm "
          f"(2.5cm 안 {int((D < 0.025).sum())}/{len(D)})", flush=True)
    At, rt, kt = fit(Xt, Yt)
    Ab, rb, kb = fit(Xb, Yb)
    print(f"위 블록 식: 평균 {100 * rt[kt].mean():.2f}cm 최대 {100 * rt[kt].max():.2f}cm / 가린 아래 블록 식: 평균 {100 * rb[kb].mean():.2f}cm 최대 {100 * rb[kb].max():.2f}cm", flush=True)
    d2 = np.linalg.norm(np.array(Xt) @ At - np.array(Xb) @ Ab, axis=1)
    print(f"새 식으로 잰 위·아래 거리: 평균 {100 * d2.mean():.2f}cm 최대 {100 * d2.max():.2f}cm (2.5cm 안 {int((d2 < 0.025).sum())}/{len(d2)})", flush=True)
    Path(a.out).write_text(json.dumps({"top": At.T.tolist(), "bottom": Ab.T.tolist()}, indent=1))
    print("저장", a.out)


if __name__ == "__main__":
    main()
