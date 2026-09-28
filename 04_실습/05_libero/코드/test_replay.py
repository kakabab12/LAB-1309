#!/usr/bin/env python
"""
궤적 재생 전문가 검증 — 전환 순간부터 **전문가가 B 를 끝까지** 한다

  전환 순간(그릇 쥔 지 3스텝) → 전문가가 바로 세워 내려놓기 → 정책의 성공 궤적을 **재생**

이 성공률이 **학습 데이터의 질**을 정한다. 전문가가 못 하면 가르칠 수도 없다.
"""
import argparse
import json

import numpy as np

import scripted_expert as se
import switch_experiment as sx
import test_redirect as tr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default="8:0,8:7,8:5,8:9,8:3")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--start-episode", type=int, default=0)
    p.add_argument("--out", default="outputs/expert/replay.json")
    a = p.parse_args()
    res = {}
    for pr in a.pairs.split(","):
        at, bt = map(int, pr.split(":"))
        ra = sx.default_args()
        ra.task_a, ra.task_b, ra.strategy = at, bt, "flush"
        runner = sx.Runner(ra)
        ok_n = tried = 0
        mins = []
        for i in range(a.start_episode, a.start_episode + a.episodes):
            ep = sx.Episode(runner, i)
            runner.policy.reset()
            if not tr.to_switch(ep, runner, at):
                ep.env.close()
                continue
            tried += 1
            t_sw = len(ep.log["pos"])
            up = se.put_down(ep)
            ok, n = se.replay(ep, bt)
            ok_n += int(ok)
            mh = se.min_home_dist(ep, t_sw)
            mins.append(mh)
            print(f"A{at}→B{bt} ep{i}: 내려놓기 {'바로 섬' if up else '기울어짐'} → 재생 {'성공' if ok else '실패'} ({n}스텝)"
                  f"  (초기 자세 최소 {100 * mh:.1f}cm)", flush=True)
            ep.env.close()
        res[pr] = {"success": int(ok_n), "tried": int(tried),
                   "min_home_cm": round(100 * min(mins), 1) if mins else None}
        print(f"  == A{at}→B{bt}: {ok_n}/{tried}  (초기 자세 최소 {100 * min(mins) if mins else float('nan'):.1f}cm)",
              flush=True)
    json.dump(res, open(a.out, "w"), indent=2, default=int)


if __name__ == "__main__":
    main()
