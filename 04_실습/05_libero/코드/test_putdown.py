#!/usr/bin/env python
"""
가설 검증 — **전문가가 깔끔하게 내려놓아 주면, 그 다음은 정책이 할 수 있는가**

  전환 순간(그릇을 쥔 지 3스텝) → 전문가가 그릇을 바로 세워 내려놓는다 → **원래 정책이** B 를 한다

  성공률이 원래 전환(정책이 스스로 떨어뜨림)보다 크게 오르면
    → 가르칠 것은 **"내려놓기" 하나**다. 그 뒤는 정책이 이미 할 줄 안다
  안 오르면
    → 내려놓기가 문제가 아니다. 교란된 자세에서 B 자체를 못 하는 것이다
"""
import argparse
import json

import numpy as np

import scripted_expert as se
import switch_experiment as sx
import test_redirect as tr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default="8:5,8:9,8:7,8:3,8:0")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--approach", action="store_true",
                   help="내려놓은 뒤 B 의 작업 대상 근처(정책 궤적 위의 한 점)까지 전문가가 데려간다")
    p.add_argument("--approach-src", choices=["fixed", "demo"], default="fixed",
                   help="fixed: approach.json 의 고정 자세 / demo: 녹화 궤적 위 첫 접촉점 근처 (물체 기준)")
    p.add_argument("--out", default="outputs/expert/putdown.json")
    a = p.parse_args()
    res = {}
    for pr in a.pairs.split(","):
        at, bt = map(int, pr.split(":"))
        ra = sx.default_args()
        ra.task_a, ra.task_b, ra.strategy = at, bt, "flush"
        runner = sx.Runner(ra)
        chk_b = sx.GoalChecker(runner.suite, bt)
        ok_n = tried = upright = 0
        mins = []
        for i in range(a.episodes):
            ep = sx.Episode(runner, i)
            runner.policy.reset()
            if not tr.to_switch(ep, runner, at):
                ep.env.close()
                continue
            tried += 1
            t_sw = len(ep.log["pos"])
            up = se.put_down(ep)
            upright += up
            if a.approach:
                (se.demo_approach if a.approach_src == "demo" else se.approach)(ep, bt)
            # 여기서부터 원래 정책이 B 를 한다
            ep.plan, ep.exec_left, ep.plan_norm = np.zeros((0, 7)), 0, None
            runner.policy.reset()
            why, n = ep.run_policy(chk_b.language, "B", a.max_steps, chk_b)
            ok = why == "success"
            ok_n += ok
            mh = se.min_home_dist(ep, t_sw)
            mins.append(mh)
            print(f"A{at}→B{bt} ep{i}: 내려놓기 {'바로 섬' if up else '기울어짐'}  →  B {'성공' if ok else '실패'}"
                  f"  (초기 자세 최소 {100 * mh:.1f}cm)", flush=True)
            ep.env.close()
        res[pr] = {"success": int(ok_n), "tried": int(tried), "upright": int(upright),
                   "min_home_cm": round(100 * min(mins), 1) if mins else None}
        print(f"  == A{at}→B{bt}: {ok_n}/{tried}  (바로 선 채 놓음 {upright}/{tried}, "
              f"초기 자세 최소 {100 * min(mins) if mins else float('nan'):.1f}cm)", flush=True)
    json.dump(res, open(a.out, "w"), indent=2, default=int)


if __name__ == "__main__":
    main()
