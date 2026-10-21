#!/usr/bin/env python
"""정면 카메라 변환식을 책상 전체로 맞추기 (10/22, 받침대 없음)
   calib_front.py 는 블록마다 따로, 장면 처음 배치(제자리 ±2cm)로만 맞췄다. 그래서 블록이 다른 곳(판 위 등)에 가면 크게 틀렸다 —
   흰 판 위의 노란 막대를 11cm 떨어진 곳으로 읽어, 한국어 명령 시연(korean_live)에서 일이 끝난 줄 몰랐다 (장면 1005).
   여기서는 블록 4개를 책상 아무 데나(판 위 포함, 방향도 아무렇게나) 옮겨 놓고 찍어, 블록 모두에 같은 식 하나를 맞춘다
   (블록은 크기가 같아 화면 위치 → 책상 위치 관계가 같다). 판은 움직이지 않으니 calib_front 값을 그대로 둔다.
   2차식에서 2cm 넘게 벗어난 점(로봇 팔에 반쯤 가린 블록)은 빼고 다시 맞춘다.
   python piper_sim/calib_front_wide.py --out piper_sim/front_calib_wide.json"""
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

BLOCKS = ["red", "green", "blue", "yellow"]


def fit(X, Y):
    X, Y = np.array(X), np.array(Y)
    keep = np.ones(len(X), bool)
    for _ in range(3):
        A, *_ = np.linalg.lstsq(X[keep], Y[keep], rcond=None)
        res = np.linalg.norm(X @ A - Y, axis=1)
        keep = res < max(0.02, 3 * np.median(res[keep]))
    return A, res, keep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", type=int, default=60)
    p.add_argument("--per-scene", type=int, default=6, help="한 장면에서 블록을 옮겨 찍는 횟수")
    p.add_argument("--out", default="piper_sim/front_calib_wide.json")
    p.add_argument("--base", default="piper_sim/front_calib.json")
    a = p.parse_args()
    rng = np.random.default_rng(22)
    X, Y, C = [], [], []
    r = so.SimRunner(8)
    ep = sx.Episode(r, 6000)          # 장면 하나를 계속 쓴다. 블록만 옮긴다
    for i in range(a.scenes):
        for _ in range(a.per_scene):
            sim = ep.inner.sim             # 옮긴 블록이 우연히 과제를 마치면 환경이 다시 시작하며 sim 을 새로 만든다 — 매번 새로 가져옴
            # 블록 4개를 서로 9cm 넘게 떨어진 아무 자리에 (로봇 밑동에서 18~42cm, 책상 앞쪽)
            pts = []
            while len(pts) < 4:
                q = np.array([rng.uniform(-0.18, 0.10), rng.uniform(-0.30, 0.30)])
                if 0.18 < np.linalg.norm(q - [-0.33, 0]) < 0.42 and all(np.linalg.norm(q - p_) > 0.09 for p_ in pts):
                    pts.append(q)
            for c, xy in zip(BLOCKS, pts):
                o = ep.inner.objects_dict[NAMES[c]]
                qp = sim.data.get_joint_qpos(o.joints[0]).copy()
                th = rng.uniform(-np.pi, np.pi)
                qp[:3] = [xy[0], xy[1], 0.93]
                qp[3:] = [np.cos(th / 2), 0, 0, np.sin(th / 2)]
                sim.data.set_joint_qpos(o.joints[0], qp)
            sim.forward()
            for _ in range(15):          # 내려앉게
                obs, *_ = ep.env.step(np.zeros(7))
            f = objdet.detect(np.asarray(obs["pixels"]["image"]), np.asarray(obs["pixels"]["image2"]))
            for k, c in enumerate(objdet.OBJECTS):
                if c in BLOCKS and f[2 * k] >= 0:
                    X.append(feats(f[2 * k], f[2 * k + 1]))
                    Y.append(ep.obj_pos(NAMES[c])[:2])
                    C.append(c)
        if i % 10 == 9:
            print(f"{i + 1}장면, 점 {len(X)}", flush=True)
    A, res, keep = fit(X, Y)
    C = np.array(C)
    print(f"블록 공통 식: 점 {len(X)} (뺀 점 {int((~keep).sum())}), 남는 오차 평균 {100 * res[keep].mean():.2f}cm 최대 {100 * res[keep].max():.2f}cm", flush=True)
    for c in BLOCKS:
        m = (C == c) & keep
        print(f"  {c}: 평균 {100 * res[m].mean():.2f}cm 최대 {100 * res[m].max():.2f}cm ({m.sum()}점)", flush=True)
    # 예전 식(블록마다, 처음 배치로 맞춘 것)이 같은 점에서 얼마나 틀리나
    old = {k: np.array(v) for k, v in json.loads(Path(a.base).read_text()).items()}
    Xa, Ya = np.array(X), np.array(Y)
    e_old = np.array([np.linalg.norm(old[c] @ Xa[j] - Ya[j]) for j, c in enumerate(C)])
    print(f"예전 식 같은 점: 평균 {100 * e_old.mean():.2f}cm, 90% {100 * np.percentile(e_old, 90):.2f}cm, 최대 {100 * e_old.max():.2f}cm", flush=True)
    out = {k: v.tolist() for k, v in old.items()}
    for c in BLOCKS:
        out[c] = A.T.tolist()
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("저장", a.out)


if __name__ == "__main__":
    main()
