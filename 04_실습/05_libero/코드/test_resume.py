#!/usr/bin/env python
"""
A 재개 시험 — B 를 마친 뒤 **원래 하던 일(A)로 돌아가는가**

  전환 순간 → 전문가가 내려놓기 + B 근처로 접근 → 정책이 B 완료
  → 지시를 다시 A 로 → 같은 상태에서 두 가지를 비교 (시뮬레이터 상태 저장·복원)
       (a) 정책 혼자
       (b) 전문가가 A 의 물체(내려놓은 그릇) 근처로 접근 → 정책

⛔ 접근 목표는 초기 자세에서 10cm 밖 (scripted_expert.MIN_HOME). 실제로 간 최소 거리도 기록한다.
"""
import argparse
import json

import numpy as np
import torch

import scripted_expert as se
import switch_experiment as sx
import test_redirect as tr


def restore(ep, state):
    ep.inner.timestep = 0
    ep.inner.done = False
    ep.obs = ep.env._format_raw_obs(ep.env._env.regenerate_obs_from_state(state))
    ep.plan, ep.exec_left, ep.plan_norm, ep.pending = np.zeros((0, 7)), 0, None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default="8:7,4:7,8:0,1:7")  # B3 은 그릇을 서랍에 넣으므로 A 재개와 충돌 — 뺐다
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--start-episode", type=int, default=0)
    p.add_argument("--out", default="outputs/expert/resume.json")
    a = p.parse_args()
    res = {}
    for pr in a.pairs.split(","):
        at, bt = map(int, pr.split(":"))
        ra = sx.default_args()
        ra.task_a, ra.task_b, ra.strategy = at, bt, "flush"
        runner = sx.Runner(ra)
        chk_a = sx.GoalChecker(runner.suite, at)
        chk_b = sx.GoalChecker(runner.suite, bt)
        r = {"b_ok": 0, "tried": 0, "alone": 0, "approach": 0, "min_home_cm": []}
        for i in range(a.start_episode, a.start_episode + a.episodes):
            ep = sx.Episode(runner, i)
            runner.policy.reset()
            if not tr.to_switch(ep, runner, at):
                ep.env.close()
                continue
            r["tried"] += 1
            se.put_down(ep)
            (se.demo_approach if bt in (5, 9) else se.approach)(ep, bt)
            ep.plan, ep.exec_left, ep.plan_norm = np.zeros((0, 7)), 0, None
            runner.policy.reset()
            why, _ = ep.run_policy(chk_b.language, "B", 300, chk_b)
            if why != "success":
                print(f"A{at}→B{bt} ep{i}: B 실패 — 재개 시험 안 함", flush=True)
                ep.env.close()
                continue
            r["b_ok"] += 1
            state = ep.env._env.get_sim_state().copy()
            out = {}
            for mode in ["alone", "approach"]:
                restore(ep, state)
                torch.manual_seed(50_000 + i)
                runner.policy.reset()
                t0 = len(ep.log["pos"])
                if mode == "approach":
                    se.demo_approach(ep, at)
                    ep.plan, ep.exec_left, ep.plan_norm = np.zeros((0, 7)), 0, None
                    runner.policy.reset()
                why, _ = ep.run_policy(chk_a.language, "A2", 300, chk_a)
                ok = why == "success"
                r[mode] += ok
                mh = se.min_home_dist(ep, t0)
                r["min_home_cm"].append(round(100 * mh, 1))
                out[mode] = f"{'성공' if ok else '실패'}(초기 {100 * mh:.0f}cm)"
            print(f"A{at}→B{bt}→A{at} ep{i}: 정책 혼자 {out['alone']}  /  접근 후 정책 {out['approach']}", flush=True)
            ep.env.close()
        res[pr] = r
        print(f"  == A{at}→B{bt}→A{at}: B 성공 {r['b_ok']}/{r['tried']} 중 재개 — 혼자 {r['alone']}, "
              f"접근 후 {r['approach']}  (초기 자세 최소 {min(r['min_home_cm'], default=float('nan'))}cm)", flush=True)
    json.dump(res, open(a.out, "w"), indent=2, default=int)


if __name__ == "__main__":
    main()
