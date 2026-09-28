#!/usr/bin/env python
"""
방향 틀기 전문가 검증 — 실제 전환 순간(그릇을 쥔 지 3스텝)에서 넘겨받는다

원래 정책으로 A 를 하다 그릇을 쥐고 3스텝 → 그 순간부터 전문가가 B 로 방향을 튼다.
원래 정책은 이 순간 '일단 놓는' 반사를 보인다.
"""
import argparse
import json

import numpy as np

import scripted_expert as se
import switch_experiment as sx


def to_switch(ep, runner, a_task):
    chk_a = sx.GoalChecker(runner.suite, a_task)
    g = {"t": None}

    def watch():
        if g["t"] is None and ep.holding():
            g["t"] = len(ep.log["pos"])

    why, _ = ep.run_policy(chk_a.language, "A", 300, chk_a,
                           lambda t: g["t"] is not None and t >= g["t"] + 3, watch)
    return why == "trigger" and sx.gripper_closed(ep.obs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default="8:1,8:4,1:8,1:4,4:8,4:1")
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--out", default="outputs/expert/redirect.json")
    a = p.parse_args()
    res = {}
    for pr in a.pairs.split(","):
        at, bt = map(int, pr.split(":"))
        ra = sx.default_args()
        ra.task_a, ra.strategy = at, "none"
        runner = sx.Runner(ra)
        ok_n = tried = 0
        for i in range(a.episodes):
            ep = sx.Episode(runner, i)
            runner.policy.reset()
            if not to_switch(ep, runner, at):
                ep.env.close()
                continue
            tried += 1
            ok, n = se.redirect(ep, bt)
            ok_n += ok
            print(f"A{at}→B{bt} ep{i}: {'성공' if ok else '실패'}  {n}스텝", flush=True)
            ep.env.close()
        res[pr] = (ok_n, tried)
        print(f"  == A{at}→B{bt}: {ok_n}/{tried}", flush=True)
    print("\n== 방향 틀기 전문가")
    for k, (o, n) in res.items():
        print(f"  A{k.replace(':', '→B')}: {o}/{n} = {100 * o / max(n, 1):.0f}%")
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
