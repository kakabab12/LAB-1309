#!/usr/bin/env python
"""
스크립트 전문가가 **정말 잘 집는가** — 가르치기 전에 선생부터 검증한다

세 조건 (쉬운 것 → 진짜 시험)
  ① 정상 시작       교란 없이. 여기서 안 되면 스크립트 자체가 틀렸다
  ② 27cm 교란       팔을 전환 시점만큼 옮겨 놓고 시작. 원래 정책은 T8 에서 11%
  ③ ⭐ 전환 뒤       원래 정책으로 A 를 하다 그릇을 쥔 순간 → 원래 정책이 전환에서 하듯
                    그릇을 놓게 두고 → **그 상태에서** 전문가가 그릇을 다시 집어 접시에 놓는다.
                    원래 정책은 이 상황에서 그릇으로 돌아가지도 못한다 (놓은 뒤 16cm 밖)

  ③ 이 높아야 이 전문가의 시범이 쓸모가 있다.
"""
import argparse
import json

import numpy as np
from scipy.spatial.transform import Rotation

import scripted_expert as se
import switch_experiment as sx


def perturb(ep, off_cm, rng):
    d = rng.normal(size=3)
    d[2] = abs(d[2]) * 0.5
    d /= np.linalg.norm(d)
    tgt = ep.home[0] + d * off_cm / 100.0
    se.servo(ep, tgt, ep.home[1].copy(), -1.0, max_steps=150, tol=0.01, vmax=1.0, phase="R")


def after_switch(ep, runner, a_task, b_task):
    """원래 정책으로 A 를 하다 쥐고 3스텝 → B 지시로 원래 정책을 돌려 **그릇을 놓게** 둔다."""
    chk_a = sx.GoalChecker(runner.suite, a_task)
    chk_b = sx.GoalChecker(runner.suite, b_task)
    g = {"t": None}

    def watch():
        if g["t"] is None and ep.holding():
            g["t"] = len(ep.log["pos"])

    def trig(t):
        return g["t"] is not None and t >= g["t"] + 3

    why, _ = ep.run_policy(chk_a.language, "A", 300, chk_a, trig, watch)
    if why != "trigger":
        return False
    ep.plan, ep.exec_left = np.zeros((0, 7)), 0
    # 원래 정책이 전환에서 하는 대로 — 그릇을 놓을 때까지 (최대 80스텝)
    for _ in range(80):
        ep.step(ep.policy_action(chk_b.language), "B")
        if not sx.gripper_closed(ep.obs):
            break
    for _ in range(20):  # 놓인 그릇이 안정될 시간
        ep.step(sx.get_libero_dummy_action(), "B")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=int, default=8)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--cond", choices=["normal", "perturb", "switch"], default="normal")
    p.add_argument("--offset-cm", type=float, default=27)
    p.add_argument("--a-task", type=int, default=8, help="switch 조건: 원래 하던 일")
    p.add_argument("--b-task", type=int, default=3, help="switch 조건: 그릇을 놓게 만드는 새 지시")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    ra = sx.default_args()
    ra.task_a, ra.strategy, ra.max_steps = (a.a_task if a.cond == "switch" else a.task), "none", 300
    runner = sx.Runner(ra)
    rows = []
    for i in range(a.episodes):
        ep = sx.Episode(runner, i)
        runner.policy.reset()
        rng = np.random.default_rng(5000 + i)
        if a.cond == "perturb":
            perturb(ep, a.offset_cm, rng)
        elif a.cond == "switch":
            if not after_switch(ep, runner, a.a_task, a.b_task):
                print(f"ep{i}: 전환 시점에 못 닿음", flush=True)
                ep.env.close()
                continue
        obj = se.TASKS[a.task][0]
        start = sx.eef_pos(ep.obs)
        dist0 = 100 * float(np.linalg.norm(start - ep.obj_pos(obj)))
        ok, n = se.pick_place(ep, a.task)
        rows.append({"episode": i, "success": ok, "steps": n, "start_dist_cm": dist0})
        print(f"ep{i}: {'성공' if ok else '실패'}  {n}스텝  (시작 때 물체까지 {dist0:.1f}cm)", flush=True)
        ep.env.close()

    k = sum(r["success"] for r in rows)
    print(f"\n== 스크립트 전문가 [{a.cond}]  T{a.task}:  {k}/{len(rows)} = {100 * k / max(len(rows), 1):.0f}%")
    if a.out:
        json.dump({"args": vars(a), "rows": rows}, open(a.out, "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
