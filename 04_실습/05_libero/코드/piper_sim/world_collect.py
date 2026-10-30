#!/usr/bin/env python
"""여러 모양·N개 장면 시범 모으기 (10/24). 장면마다 일 k 개(설정 범위에서 뽑음)를 끼어들기 겹쳐서(또는 차례로) 하고,
   지시가 바뀔 때마다 구간을 잘라 구간마다 그때의 지시문으로 저장한다 (switch_v6 의 SA/SB/SR 와 같은 생각, 깊이 제한 없음).
   학습용 장면은 2000번대 이상만 (1000번대는 평가).
   PIPER_NOPED=1 MUJOCO_GL=egl python piper_sim/world_collect.py --seeds 3000-3099 --k 1 4 --nested-frac 0.7 --dart 0.1 --out data/world_default"""
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
import collect_expert as ce  # noqa: E402
import piper_sim.world_expert as WE  # noqa: E402
import scripted_expert as se  # noqa: E402
import switch_experiment as sx  # noqa: E402


class SegRec:
    """지시문이 바뀔 때마다 새 기록기. (지시문, Rec) 목록."""

    def __init__(self):
        self.segs = []

    def say(self, lang):
        if not self.segs or self.segs[-1][0] != lang:
            self.segs.append((lang, ce.Rec()))

    def __call__(self, o, a):
        self.segs[-1][1](o, a)


def rng_list(s):
    out = []
    for part in s.split(","):
        a, b = part.split("-") if "-" in part else (part, part)
        out += list(range(int(a), int(b) + 1))
    return [i for i in out if i >= 2000]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", required=True)
    p.add_argument("--k", type=int, nargs=2, default=[1, 4], help="일 개수 범위 (양 끝 포함)")
    p.add_argument("--nested-frac", type=float, default=0.7, help="끼어들기 겹치기 비율 (나머지는 차례로)")
    p.add_argument("--dart", type=float, default=0.1)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    se.DART_SIGMA = a.dart
    out = Path(a.out) / "episodes"
    out.mkdir(parents=True, exist_ok=True)
    st = {"tried": 0, "ok": 0, "segs": 0, "frames": 0, "by_k": {}}
    t0 = time.time()
    for seed in rng_list(a.seeds):
        if list(out.glob(f"W{seed}_*.npz")):
            continue
        rng = np.random.default_rng(seed)
        se._rng = np.random.default_rng(seed + 7)
        k = int(rng.integers(a.k[0], a.k[1] + 1))
        nested = bool(k > 1 and rng.random() < a.nested_frac)
        lo, hi = W.CFG["n_objects"]
        sc = W.sample_scene(seed, n_objects=int(rng.integers(max(lo, k), max(hi, k) + 1)))
        tasks = W.chain(sc, k, rng)
        rec = SegRec()
        ep = WE.make_episode(sc, seed, tasks)
        mv = W.sample_motion(seed, k)                 # 10/30 동작 변수 (설정 motion): 끼어드는 때·속도·잡음
        ep.motion = mv
        try:
            ok, each = WE.run_chain(ep, tasks, nested=nested, record=rec, say=rec.say, switch=mv["switch"] if mv else None)
        except Exception as e:
            ok, each = False, []
            print(f"  예외 {seed}: {type(e).__name__}: {e}"[:200], flush=True)
        st["tried"] += 1
        kk = st["by_k"].setdefault(f"{k}{'n' if nested else 's'}", [0, 0])
        kk[1] += 1
        if ok:
            kk[0] += 1
            st["ok"] += 1
            for j, (lang, r) in enumerate(rec.segs):
                if len(r) < 10:
                    continue
                r.save(out / f"W{seed}_k{k}{'n' if nested else 's'}_s{j}.npz", lang, source="world_expert",
                       scene=json.dumps({"seed": seed, "objects": sc.objects, "pads": sc.pads,
                                         "env_info": getattr(sc, "env_info", {}), "motion": mv}), seg=j)
                st["segs"] += 1
                st["frames"] += len(r)
        print(f"{seed} k={k} {'겹침' if nested else '차례'} 물체 {len(sc.objects)} 판 {len(sc.pads)}: {'성공' if ok else '실패'} "
              f"구간 {len(rec.segs)} 스텝 {len(ep.acts)} ({time.time() - t0:.0f}초)", flush=True)
        ep.env.close()
    st["wall_sec"] = round(time.time() - t0, 1)
    print("STATS", json.dumps(st, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
