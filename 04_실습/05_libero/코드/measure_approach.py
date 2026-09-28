#!/usr/bin/env python
"""
각 태스크의 **접근 자세** — 정책이 성공할 때, 작업 대상 근처에 도착한 순간의 손 자세

왜 필요한가 (2026-09-28)
  정책은 **교란된 자세에서** 무너진다 (27cm 교란에서 물체 집기 4%).
  그런데 **자기가 원래 성공하던 궤적 위**에 있으면 잘 한다.
  → 전문가가 손을 **그 궤적 위의 한 점(작업 대상 근처)** 까지 자연스럽게 데려가고,
    거기서부터 정책이 마무리하게 한다.

  이 점은 **초기 자세가 아니라 작업 대상 근처**다 → 초기 자세 복귀 금지 제약을 지킨다.

어떻게 정하나
  정책이 성공한 궤적에서, 손이 **마지막 위치까지 12cm 안으로 처음 들어온 순간** 의 자세.
  (작업 대상에 "도착했다"고 볼 수 있는 지점. 너무 가까우면 이미 조작 중이다)
  여러 번 성공한 것의 평균을 쓴다.
"""
import argparse
import json

import numpy as np
from scipy.spatial.transform import Rotation

import switch_experiment as sx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=int, nargs="+", default=[0, 3, 5, 7, 9])
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--radius", type=float, default=0.12)
    p.add_argument("--out", default="outputs/expert/approach.json")
    a = p.parse_args()
    res = {}
    for task in a.tasks:
        ra = sx.default_args()
        ra.task_a, ra.strategy = task, "none"
        runner = sx.Runner(ra)
        chk = runner.chk_a
        poses, rots, fracs = [], [], []
        for i in range(a.episodes):
            ep = sx.Episode(runner, i)
            runner.policy.reset()
            P, R = [], []
            ok = False
            for _ in range(300):
                ep.step(ep.policy_action(chk.language), "A")
                P.append(sx.eef_pos(ep.obs).copy())
                R.append(Rotation.from_matrix(ep.obs["robot_state"]["eef"]["mat"]).as_rotvec())
                if chk(ep.env):
                    ok = True
                    break
            if ok and len(P) > 10:
                P = np.array(P)
                d = np.linalg.norm(P - P[-1], axis=1)
                k = int(np.argmax(d < a.radius))
                poses.append(P[k]); rots.append(R[k]); fracs.append(k / len(P))
            print(f"T{task} ep{i}: {'성공' if ok else '실패'}", flush=True)
            ep.env.close()
        if poses:
            Pm = np.mean(poses, 0)
            Rm = Rotation.from_rotvec(rots).mean().as_rotvec()
            res[task] = {"language": chk.language, "n": len(poses),
                         "pos": Pm.tolist(), "pos_std": np.std(poses, 0).tolist(),
                         "rotvec": Rm.tolist(), "frac_of_traj": float(np.mean(fracs))}
            print(f"  → T{task}: 접근 자세 {np.round(100 * Pm, 1)}cm (퍼짐 {np.round(100 * np.std(poses, 0), 1)}) "
                  f"— 궤적의 {100 * np.mean(fracs):.0f}% 지점", flush=True)
    json.dump(res, open(a.out, "w"), indent=2, ensure_ascii=False)
    print("저장:", a.out)


if __name__ == "__main__":
    main()
