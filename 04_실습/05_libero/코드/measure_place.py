#!/usr/bin/env python
"""
정책이 **성공할 때** 물체를 어디에 놓는가 — 스크립트 전문가의 놓는 목표

왜 필요한가 (2026-09-28)
  "그릇을 스토브에" 의 스토브는 물체가 아니라 **가구의 한 영역**이라 위치를 바로 얻기 어렵다.
  추측하지 말고 **원래 정책이 성공할 때 그릇이 어디 놓였는지** 재서 그대로 쓴다.
  집는 자세도 같은 방식으로 쟀다 (`measure_grasp_pose.py`).

무엇을 재나
  태스크가 성공한 순간
    ① 물체 위치 (월드)          → 놓는 목표
    ② 그리퍼가 열리던 순간의 손 위치 − 물체 위치  → 놓을 때 손 자세
"""
import argparse
import json

import numpy as np

import switch_experiment as sx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--obj", default="akita_black_bowl_1")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--out", default="outputs/expert/place.json")
    a = p.parse_args()

    res = {}
    ra = sx.default_args()
    ra.strategy, ra.max_steps = "none", 300
    for task in a.tasks:
        ra.task_a = task
        runner = sx.Runner(ra)
        chk = runner.chk_a
        places, rel_hand = [], []
        for i in range(a.episodes):
            ep = sx.Episode(runner, i)
            runner.policy.reset()
            release_hand = None
            was_closed = False
            for t in range(300):
                ep.step(ep.policy_action(chk.language), "A")
                closed = sx.gripper_closed(ep.obs)
                if was_closed and not closed and release_hand is None:
                    release_hand = sx.eef_pos(ep.obs) - ep.obj_pos(a.obj)
                was_closed = closed
                if chk(ep.env):
                    places.append(ep.obj_pos(a.obj).tolist())
                    if release_hand is not None:
                        rel_hand.append(release_hand.tolist())
                    break
            print(f"T{task} ep{i}: {'성공' if chk(ep.env) else '실패'}", flush=True)
            ep.env.close()
        if places:
            P = np.array(places)
            res[task] = {"language": chk.language, "n": len(P),
                         "place_mean": P.mean(0).tolist(), "place_std": P.std(0).tolist(),
                         "hand_rel_mean": (np.mean(rel_hand, 0).tolist() if rel_hand else None)}
            print(f"  → T{task} '{chk.language}': 놓인 위치 {np.round(100 * P.mean(0), 1)}cm "
                  f"(퍼짐 {np.round(100 * P.std(0), 1)})", flush=True)
    json.dump(res, open(a.out, "w"), indent=2, ensure_ascii=False)
    print("저장:", a.out)


if __name__ == "__main__":
    main()
