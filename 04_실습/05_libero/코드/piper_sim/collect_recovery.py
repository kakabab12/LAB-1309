#!/usr/bin/env python
"""블록 장면 회복 시범 (10/6 새벽): 학생 모델이 실제로 하는 실패 — 블록 옆 1~3cm 에서 오므려 빈손, 손가락이 윗면에 걸림 —
을 시범 프로그램이 녹화 없이 만든 뒤, 그 상태에서부터 '다시 열고, 맞추고, 집어 옮기기'만 녹화한다.
실패 동작은 녹화하지 않으므로 학생이 실패를 따라 하지 않는다 (DAgger 의 넘겨받은 뒤 구간과 같은 꼴).
파일 이름은 DF{과제}_ep{장면} — 'D' 로 시작해 학습에서 교정 시범 몫(--dagger-frac)으로 뽑힌다.

  PIPER_BLOCKS=1 python piper_sim/collect_recovery.py --tasks 0 1 2 3 4 5 6 7 8 9 --episodes 2600-2649 --out data/piper_blk_recovery
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import piper_sim.piper_robot as pr  # noqa: E402

pr.use_piper_in_lerobot()
import collect_expert as ce  # noqa: E402
import piper_sim.blocks as pb  # noqa: E402
import scripted_expert as se  # noqa: E402
import sim_only as so  # noqa: E402
import switch_experiment as sx  # noqa: E402
import task_experts as te  # noqa: E402
from collect_v6 import ep_range, settle  # noqa: E402


def fake_miss(ep, obj, rng):
    """녹화 없이: 블록 옆 1.2~3cm 로 가서 내려가 오므린다 (빈손이거나 모서리만 걸림). 절반은 오므린 채 조금 들어 올린다."""
    a, b, h = te.block_size(obj)
    o = ep.obj_pos(obj)
    g = o + [0, 0, min(0.025, h / 2)]
    cands = te.block_grasp_candidates(ep, obj)
    gm = cands[int(rng.integers(len(cands)))][1]
    ang = rng.uniform(0, 2 * np.pi)
    r = rng.uniform(0.012, 0.035)
    off = np.array([r * np.cos(ang), r * np.sin(ang), 0.0])
    se.servo(ep, g + off + [0, 0, 0.08], gm, -1.0, tol=0.02, vmax=0.5, max_steps=80)
    se.servo(ep, g + off + [0, 0, rng.uniform(0.0, 0.02)], gm, -1.0, tol=0.006, vmax=0.3, max_steps=40)
    se.hold(ep, 1.0, 10)
    if rng.random() < 0.5:
        se.servo(ep, sx.eef_pos(ep.obs) + [0, 0, rng.uniform(0.03, 0.08)], gm, 1.0, tol=0.02, max_steps=25)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--episodes", required=True)
    p.add_argument("--dart", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    se._rng = np.random.default_rng(a.seed + 1)
    out = Path(a.out) / "episodes"
    out.mkdir(parents=True, exist_ok=True)
    stats = {"tried": 0, "saved": 0, "frames": 0}
    t0 = time.time()
    for task in a.tasks:
        obj = pb.TASKS[task][0]
        r = so.SimRunner(task)
        k = 0
        for i in ep_range(a.episodes):
            f = out / f"DF{task}_ep{i}.npz"
            if f.exists():
                k += 1
                continue
            ep = sx.Episode(r, i)
            ep.inner.horizon = 4000
            try:
                se.DART_SIGMA = 0.0
                fake_miss(ep, obj, rng)
                se.DART_SIGMA = a.dart
                rec = ce.Rec()
                if ep.finger_object() == obj:
                    # 빗나가게 쥐었는데 블록이 걸려 잡혔다 → 놓지 않고, 어긋나게 쥔 그대로 어긋남을 재서 맞춰 놓는다
                    #   (놓았다 다시 집는 시범은 '쥐자마자 놓는' 버릇을 만든다 — Panda v6c)
                    se.hold(ep, 1.0, 4, rec)
                    te.block_carry(ep, task, rec)
                    stats["carry"] = stats.get("carry", 0) + 1
                else:
                    te.EXPERT[task](ep, rec)
                ok = settle(ep, r.chk_a) and se.min_home_dist(ep, 0) >= 0.0
            except Exception as e:
                print(f"  예외 T{task} ep{i}: {type(e).__name__}: {e}"[:160], flush=True)
                ok = False
            stats["tried"] += 1
            if ok and len(rec) > 10:
                rec.save(f, r.chk_a.language, source="recovery")
                stats["saved"] += 1
                stats["frames"] += len(rec)
                k += 1
            ep.env.close()
        print(f"  == T{task}: {k} 저장", flush=True)
    stats["wall_sec"] = round(time.time() - t0, 1)
    print("STATS", json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
