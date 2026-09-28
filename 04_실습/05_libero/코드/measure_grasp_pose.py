#!/usr/bin/env python
"""
정책이 **성공적으로 집을 때** 물체에 대해 손이 어디에 어떤 방향으로 있는가

왜 필요한가 (2026-09-28)
  스크립트 전문가를 만들어 "어디서 시작하든 물체로 가서 집기"를 시범 보이게 하려 한다.
  그런데 그릇은 속이 빈 물체라 **가운데를 위에서 집으면 안 되고 테두리를 집어야** 한다.
  어디를 어떤 방향으로 집는지 추측하지 말고, **원래 정책이 성공할 때를 재서** 그대로 쓴다.

무엇을 재나
  교란 없이 정책을 돌려, 그리퍼가 닫혀 물체가 들린 순간에
    ① 손 위치 − 물체 위치  (물체 기준 오프셋, 월드 좌표)
    ② 손 방향 (초기 방향 대비 회전)
  를 기록한다.

쓰는 법
  python measure_grasp_pose.py --task 8 --obj akita_black_bowl_1 --episodes 10
"""
import argparse
import json

import numpy as np
from scipy.spatial.transform import Rotation

import switch_experiment as sx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--obj", required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    ra = sx.default_args()
    ra.task_a, ra.strategy, ra.max_steps = a.task, "none", 300
    runner = sx.Runner(ra)
    chk = runner.chk_a
    rows = []
    for i in range(a.episodes):
        ep = sx.Episode(runner, i)
        runner.policy.reset()
        obj0 = ep.obj_pos(a.obj).copy()
        rec = None
        for t in range(300):
            ep.step(ep.policy_action(chk.language), "A")
            # 물체가 2cm 넘게 들렸고 그리퍼가 닫혀 있으면 = 성공적으로 집은 순간
            if rec is None and sx.gripper_closed(ep.obs) and ep.obj_pos(a.obj)[2] > obj0[2] + 0.02:
                hand = sx.eef_pos(ep.obs)
                obj = ep.obj_pos(a.obj)
                R = Rotation.from_matrix(ep.obs["robot_state"]["eef"]["mat"])
                H = Rotation.from_matrix(ep.home[1])
                rec = {"episode": i, "t": t,
                       "offset": (hand - obj).tolist(),
                       "obj_start": obj0.tolist(),
                       "rot_from_home_deg": float(np.degrees(np.linalg.norm((R * H.inv()).as_rotvec()))),
                       "rotvec": R.as_rotvec().tolist()}
            if chk(ep.env):
                break
        ok = bool(chk(ep.env))
        if rec:
            rec["success"] = ok
            rows.append(rec)
            print(f"ep{i}: 집음 t={rec['t']:3d}  오프셋 {np.round(100 * np.array(rec['offset']), 1)}cm  "
                  f"회전 {rec['rot_from_home_deg']:.1f}도  성공={ok}", flush=True)
        else:
            print(f"ep{i}: 못 집음  성공={ok}", flush=True)
        ep.env.close()

    good = [r for r in rows if r["success"]]
    if good:
        off = np.array([r["offset"] for r in good])
        print(f"\n== 성공한 집기 {len(good)}개의 물체 기준 오프셋 (cm)")
        print(f"   평균  {np.round(100 * off.mean(0), 1)}")
        print(f"   퍼짐  {np.round(100 * off.std(0), 1)}")
        print(f"   회전  {np.mean([r['rot_from_home_deg'] for r in good]):.1f}도 (초기 방향 대비)")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
