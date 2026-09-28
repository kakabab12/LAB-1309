#!/usr/bin/env python
"""
정책이 **원래 성공한 조작**을 녹화한다 — 궤적 재생 전문가의 재료

왜 (2026-09-28)
  서랍 열기·스토브 켜기·접시 밀기·와인병 옮기기를 전부 손으로 짜는 대신,
  정책이 **교란 없이 성공한 궤적**을 녹화해 두고 필요할 때 재생한다.
  정책은 교란 없는 상태에서는 이 조작들을 잘 한다 (서랍 60~80%, 스토브 100%, 밀기 100%).
  문제는 **교란된 자세에서 시작할 때**뿐이다 → 전문가가 궤적 시작점까지 데려가고 그대로 따라가면 된다.

무엇을 저장하나 (성공한 에피소드만)
  매 스텝: 손 위치·방향, 실행한 동작(그리퍼 포함), **모든 물체의 위치**
  → 물체가 움직였을 때 궤적을 얼마나 옮길지 계산하는 데 쓴다
"""
import argparse
import pickle

import numpy as np
from scipy.spatial.transform import Rotation

import switch_experiment as sx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=int, nargs="+", default=[0, 3, 5, 7, 9])
    p.add_argument("--episodes", type=int, default=15)
    p.add_argument("--start-episode", type=int, default=100, help="평가 에피소드와 겹치지 않게")
    p.add_argument("--out", default="outputs/expert/demos.pkl")
    p.add_argument("--append", action="store_true", help="기존 파일에 태스크를 더한다")
    a = p.parse_args()

    import os
    demos = pickle.load(open(a.out, "rb")) if a.append and os.path.exists(a.out) else {}
    for task in a.tasks:
        ra = sx.default_args()
        ra.task_a, ra.strategy = task, "none"
        runner = sx.Runner(ra)
        chk = runner.chk_a
        demos[task] = []
        for i in range(a.start_episode, a.start_episode + a.episodes):
            ep = sx.Episode(runner, i)
            runner.policy.reset()
            P, R, A, O = [], [], [], []
            ok = False
            for _ in range(300):
                act = np.asarray(ep.policy_action(chk.language), np.float32)
                P.append(sx.eef_pos(ep.obs).copy())
                R.append(Rotation.from_matrix(ep.obs["robot_state"]["eef"]["mat"]).as_rotvec())
                A.append(act)
                O.append({k: np.array(v) for k, v in ep.object_layout().items()})
                ep.step(act, "A")
                if chk(ep.env):
                    ok = True
                    break
            if ok:
                objs = list(O[0].keys())
                demos[task].append({
                    "episode": i, "pos": np.array(P), "rotvec": np.array(R), "action": np.array(A),
                    "objs": {k: np.stack([o[k] for o in O]) for k in objs},
                    "home_pos": ep.home[0].copy(), "home_rot": Rotation.from_matrix(ep.home[1]).as_rotvec()})
            print(f"T{task} ep{i}: {'성공 — 녹화' if ok else '실패'}", flush=True)
            ep.env.close()
        print(f"  == T{task}: 성공 궤적 {len(demos[task])}개", flush=True)
    pickle.dump(demos, open(a.out + ".tmp", "wb"))
    os.replace(a.out + ".tmp", a.out)      # 다른 실험이 읽는 중이어도 깨지지 않게
    print("저장:", a.out)


if __name__ == "__main__":
    main()
