#!/usr/bin/env python
"""
6차 학습 데이터 — **전부 전문가 방식** (2026-10-01)

왜
  5차는 [스크립트 동작]과 [원래 모델 동작]이 같은 지시문에 섞여, 실제로 돌리면 두 방식 사이를
  오가다 무너졌다 (10개 태스크 75% → 39%). 6차는 정상 수행·전환·재개를 모두 같은 전문가
  (task_experts.EXPERT, 10개 태스크 합계 98.7%)로 만든다.

모델 없이 시뮬레이터만 돌린다 (CPU). 평가용 장면(에피소드 20~49)은 쓰지 않는다.

쓰는 법
  python collect_v6.py normal --tasks 0 1 --episodes 0-19,50-139 --dart 0.1 --out data/v6_normal
  python collect_v6.py switch --pairs 8:0,8:3 --episodes 0-19,50-79 --dart 0.1 --out data/v6_switch
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import collect_expert as ce
import scripted_expert as se
import sim_only as so
import switch_experiment as sx
import task_experts as te

EVAL = set(range(20, 50))


def ep_range(s):
    out = []
    for part in s.split(","):
        a, b = part.split("-") if "-" in part else (part, part)
        out += list(range(int(a), int(b) + 1))
    return [i for i in out if i not in EVAL]


def settle(ep, chk, n=15):
    for _ in range(n):
        if chk(ep.env):
            return True
        ep.step(np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32), "E")
    return bool(chk(ep.env))


def run_normal(a, out, stats):
    for task in a.tasks:
        r = so.SimRunner(task)
        k = 0
        for i in ep_range(a.episodes):
            if (out / f"N{task}_ep{i}.npz").exists():      # 이미 모은 것은 건너뛴다 (세션이 끊겨도 이어서)
                k += 1
                continue
            ep = sx.Episode(r, i)
            rec = ce.Rec()
            te.EXPERT[task](ep, rec)
            ok = settle(ep, r.chk_a)
            if ok and len(rec) > 10:
                rec.save(out / f"N{task}_ep{i}.npz", r.chk_a.language, source="expert_normal")
                stats["frames"] += len(rec)
                k += 1
            stats["tried"] += 1
            ep.env.close()
        stats["per"][f"T{task}"] = k
        print(f"  == T{task}: {k} 저장", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["normal", "switch"])
    p.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--pairs", default="")
    p.add_argument("--episodes", default="0-19,50-139")
    p.add_argument("--dart", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    se.DART_SIGMA = a.dart
    se._rng = np.random.default_rng(a.seed)
    out = Path(a.out) / "episodes"
    out.mkdir(parents=True, exist_ok=True)
    stats = {"tried": 0, "frames": 0, "per": {}}
    t0 = time.time()
    if a.mode == "normal":
        run_normal(a, out, stats)
    else:
        import switch_v6
        switch_v6.run(a, out, stats)
    stats["wall_sec"] = round(time.time() - t0, 1)
    print("STATS", json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
