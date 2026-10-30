#!/usr/bin/env python
"""다듬기 쥐기 시험대 (10/31, VLA 없이 CPU 로): 평가 장면에서 손을 지시한 물체 위(옆 hover cm, 윗면 위 9cm)로 시범 프로그램이 옮겨 두고,
   지연 걸린 VLA 처럼 '잠깐 떠 있다(5) 천천히 내려가다(10) 빨라지며 먼저 오므리고(10) 들고(30) 놓을 곳 쪽으로 떠나는' 가짜 동작을 다듬기에 넣어 실제로 쥐는지 본다.
   위 카메라 찾기에 그 물체만 err cm 치우침을 계속 넣을 수 있다 (팔에 가린 덩어리·조명 탓 기억 오차, 기본 1026: 1.6cm 틀린 기억에 맞춰
   내려가다 손가락이 원기둥 윗면에 걸림). 화면은 OSMesa 로 그려 GPU 를 쓰지 않는다 (GPU 일꾼 수에 안 들어감).
   PIPER_NOPED=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PIPER_WORLD=piper_sim/worlds/default.json \
       python piper_sim/refine_bench.py --seeds 1026,1001 --parts grasp,place,...,gq --err 0 1.5 --out outputs/piper/bench_rb.json"""
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
import piper_sim.world_refine as WR  # noqa: E402
import scripted_expert as se  # noqa: E402
import switch_experiment as sx  # noqa: E402

DIRS = {"px": (1, 0), "mx": (-1, 0), "py": (0, 1), "my": (0, -1)}


def scene(seed, k=1):
    rng = np.random.default_rng(seed * 31 + k)                      # world_eval.main 과 같은 장면·일
    lo, hi = W.CFG["n_objects"]
    n = int(rng.integers(max(lo, k), max(hi, k) + 1))
    sc = W.sample_scene(seed, n_objects=n, n_pads=W.pick_n_pads(rng, k))
    return sc, W.chain(sc, k, rng)


def trial(seed, parts, err_cm, d, hover_cm, steps):
    sc, tasks = scene(seed)
    o, dst, lang = tasks[0]
    ep = WE.make_episode(sc, seed, tasks)
    ep.motion = W.sample_motion(seed, 1)
    R = WR.make(tuple(parts))
    bias = np.array(DIRS[d], dtype=np.float64) * err_cm / 100 if err_cm else np.zeros(2)
    loc = R.det.locate

    def locate(img, s, robot_q=None):                                  # 위 카메라 찾기에 그 물체만 치우침
        f = loc(img, s, robot_q=robot_q)
        if o in f:
            x, y, n = f[o]
            f[o] = (x + bias[0], y + bias[1], n)
        return f
    R.det.locate = locate
    zero = np.zeros(7, dtype=np.float32)
    zero[6] = -1.0
    orig = sx.Episode.step

    def watch(self, action, phase):                                  # 시범 프로그램이 움직이는 동안 다듬기는 보기만 (기억 만들기)
        orig(self, action, phase)
        R(self, zero, lang)
    sx.Episode.step = watch
    try:
        p0 = ep.obj_pos(o).copy()
        top = W.TABLE_Z + W.height(o)
        WE.approach_until(ep, o, 1.0)                                   # 시범 프로그램이 물체 위 안전 높이로
        u = np.array(DIRS[["px", "py", "mx", "my"][seed % 4]], dtype=np.float64)   # VLA 가 비껴 온 방향 (장면마다 다르게)
        tgt = np.array([p0[0] + u[0] * hover_cm / 100, p0[1] + u[1] * hover_cm / 100, top + 0.09])
        for _ in range(3):
            se.servo(ep, tgt, ep.obs["robot_state"]["eef"]["mat"].copy(), -1.0, max_steps=150, tol=0.005, vmax=0.3)
            if np.linalg.norm(np.asarray(ep.obs["robot_state"]["eef"]["pos"])[:2] - tgt[:2]) < 0.008:
                break
    finally:
        sx.Episode.step = orig
    h0 = np.asarray(ep.obs["robot_state"]["eef"]["pos"], dtype=np.float64)
    start = dict(hand_obj_cm=round(100 * float(np.linalg.norm(h0[:2] - p0[:2])), 1), above_top_cm=round(100 * float(h0[2] - top), 1))
    z_rest = float(ep.obj_pos(o)[2])
    lifted, held_any, other = False, 0, 0
    for t in range(steps):
        a = np.zeros(7, dtype=np.float32)                             # 따라가기 기본 1026 의 VLA 명령 꼴 (지연 7)
        if t < 5:
            a[2], a[6] = 0.0, -1.0                                    # 물체 위에 잠깐 떠 있음
        elif t < 15:
            a[2], a[6] = -0.3, -1.0                                   # 천천히 내려감
        elif t < 25:
            a[2], a[6] = -0.8, 1.0                                    # 빨라지며 먼저 오므림
        elif t < 55:
            a[2], a[6] = 0.6, 1.0                                     # 쥐었다고 보고 듦
        else:
            a[0], a[1], a[6] = 0.4, 0.0, 1.0                          # 놓을 곳 쪽으로 떠남
        out = R(ep, a, lang)
        ep.step(out, "T0")
        fo = ep.finger_object() if ep.holding() else None
        if fo == o:
            held_any += 1
            if float(ep.obj_pos(o)[2]) > z_rest + 0.03:
                lifted = True
        elif fo:
            other += 1
    res = dict(seed=seed, obj=o, err_cm=err_cm, dir=d, hover_cm=hover_cm, start=start, lifted=lifted, held_steps=held_any, other_steps=other,
               tilt=round(float(WE.tilt_deg(ep, o)), 1), stat={k: v for k, v in R.stat.items() if v})
    ep.env.close()
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", required=True)
    p.add_argument("--parts", required=True)
    p.add_argument("--err", type=float, nargs="+", default=[0.0, 1.5])
    p.add_argument("--dirs", nargs="+", default=list(DIRS))
    p.add_argument("--hover", type=float, default=1.5)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    done = json.loads(out.read_text()) if out.exists() else []
    have = {(r["seed"], r["err_cm"], r["dir"]) for r in done}
    t0 = time.time()
    for seed in [int(s) for s in a.seeds.split(",")]:
        for e in a.err:
            for d in (a.dirs if e else ["px"]):
                if (seed, e, d) in have:
                    continue
                r = trial(seed, a.parts.split(","), e, d, a.hover, a.steps)
                done.append(r)
                out.write_text(json.dumps(done, ensure_ascii=False, indent=1))
                print(f"{seed} {r['obj'][2:]} 시작 {r['start']} 치우침 {e}cm {d}: {'들어 올림' if r['lifted'] else '못 쥠'} 쥔 {r['held_steps']} 다른 {r['other_steps']} "
                      f"기울기 {r['tilt']} {r['stat']} ({time.time() - t0:.0f}초)", flush=True)
    n = len(done)
    print("SUMMARY", f"{sum(r['lifted'] for r in done)}/{n}", flush=True)


if __name__ == "__main__":
    main()
