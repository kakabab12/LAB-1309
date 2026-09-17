#!/usr/bin/env python
"""
오라클 Best-of-N — "좋은 후보가 존재하기는 하는가?"

배경: 학습한 판정기는 후보를 못 골랐다 (AUC V 0.9318 vs Q 0.9321, 차이 0.0003).
      후보 선택(bon/vbon/DSRL) 방향을 계속 파려면, 먼저 **고를 만한 후보가 있는지**부터 확인해야 한다.

방법: 전환 시점의 시뮬레이터 상태를 저장하고(`get_sim_state`), 그 상태에서 N번 다시 굴린다
      (`regenerate_obs_from_state` 로 복원 + flow matching 노이즈 시드만 바꿈).
      정책도 바뀌지 않고 상태도 같으므로, 차이는 **생성 노이즈뿐**이다.

⚠️ 한계: 상태를 복원할 때 MuJoCo 물리 상태는 되돌려도 **로봇 제어기(OSC)의 내부 목표값은
   되돌려지지 않는다.** 그래서 후보의 실행 순서가 결과에 조금 영향을 줄 수 있고,
   "시드 효과"와 "순서 효과"가 이 설계에서는 분리되지 않는다.
   → 순서 없이 한 번씩만 돌리는 검증 실험(--switch-noise-seed, 미사용 에피소드)으로 가려야 한다.

읽는 법:
  기본 성공률 p   후보 하나를 그냥 실행했을 때 (= flush 와 같음)
  오라클 성공률    N개 중 하나라도 성공한 상태의 비율 → **모든 후보 선택 방법의 상한**
  독립 예측       1-(1-p)^N. 성공이 순전히 노이즈 운이라면 오라클이 여기에 가까워야 한다
  묶임(clustering) 오라클 << 독립 예측 이면, 성공/실패가 **상태에서 이미 갈린다**는 뜻
                  → 후보 중 고르기로는 못 고치고, 상태 자체를 바꿔야 한다 (= 학습 / 다른 동작)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import switch_experiment as sx


def run_state(ep, chk, instruction, seed, max_steps, state):
    """저장된 상태로 되돌리고, 주어진 시드로 B 를 끝까지 수행. (성공, 스텝) 반환."""
    # 로봇수트는 스텝 카운터가 horizon(기본 1000)을 넘으면 더 이상 step 을 받지 않는다.
    # 후보를 이어서 굴리면 카운터가 누적되므로, 상태와 함께 카운터도 되돌린다.
    ep.inner.timestep = 0
    ep.inner.done = False
    raw = ep.env._env.regenerate_obs_from_state(state)
    ep.obs = ep.env._format_raw_obs(raw)
    ep.r.policy.reset()
    ep.plan = np.zeros((0, 7))
    ep.plan_norm = None
    ep.pending = None
    ep.exec_left = 0
    ep.last_grip = float(ep.obs["robot_state"]["gripper"]["qpos"][0] > 0.02) * 2 - 1
    torch.manual_seed(seed)
    why, n = ep.run_policy(instruction, "B", max_steps, chk)
    return why == "success", n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default="8:7,4:7,1:5", help="A:B 쌍 목록")
    p.add_argument("--switch-at", default="grasp:3")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--n", type=int, default=8, help="후보(재시도) 개수")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--out", default="outputs/oracle_bon")
    p.add_argument("--policy", default="HuggingFaceVLA/smolvla_libero")
    args = p.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    pairs = [tuple(int(x) for x in s.split(":")) for s in args.pairs.split(",")]
    runner = None
    all_rows = []
    for ta, tb in pairs:
        rargs = sx.default_args()
        rargs.task_a, rargs.task_b = ta, tb
        rargs.switch_at, rargs.strategy = args.switch_at, "flush"
        rargs.max_steps, rargs.policy = args.max_steps, args.policy
        # 정책은 한 번만 올리고 체커/태스크만 갈아끼운다
        if runner is None:
            runner = sx.Runner(rargs)
        else:
            runner.args = sx.fill_defaults(rargs)
            runner.chk_a = sx.GoalChecker(runner.suite, ta)
            runner.chk_b = sx.GoalChecker(runner.suite, tb)
            kind, val = args.switch_at.split(":")
            runner.switch_kind, runner.switch_val = kind, int(val)

        for ep_idx in range(args.episodes):
            t0 = time.time()
            ep = sx.Episode(runner, ep_idx)
            runner.policy.reset()
            grasp = {"t": None}

            def watch():
                if grasp["t"] is None and ep.holding():
                    grasp["t"] = len(ep.log["pos"])

            def trigger(t):
                if runner.switch_kind == "step":
                    return t >= runner.switch_val
                return grasp["t"] is not None and t >= grasp["t"] + runner.switch_val

            why, n_a = ep.run_policy(runner.chk_a.language, "A", args.max_steps, runner.chk_a, trigger, watch)
            if why != "trigger":
                print(f"A{ta}→B{tb} ep{ep_idx}: 전환 조건 미충족 ({why}) — 제외", flush=True)
                ep.env.close()
                continue
            state = ep.env._env.get_sim_state().copy()
            held = ep.nearest_object() if sx.gripper_closed(ep.obs) else None
            ep.held = held
            succ, steps = [], []
            for k in range(args.n):
                s, nb = run_state(ep, runner.chk_b, runner.chk_b.language, 10_000 + 97 * k, args.max_steps, state)
                succ.append(bool(s)); steps.append(nb)
            row = {"task_a": ta, "task_b": tb, "episode": ep_idx, "switch_at": args.switch_at,
                   "holding": held is not None, "held_object": held,
                   "success": succ, "steps": steps, "n_success": int(sum(succ)),
                   "sec": round(time.time() - t0, 1)}
            all_rows.append(row)
            print(f"A{ta}→B{tb} ep{ep_idx}: {sum(succ)}/{args.n} 성공 "
                  f"{''.join('O' if s else '.' for s in succ)} ({row['sec']}s)", flush=True)
            ep.env.close()
            json.dump({"args": vars(args), "rows": all_rows}, open(Path(args.out) / "oracle.json", "w"),
                      ensure_ascii=False, indent=2)

    # ---- 집계 ----
    N = args.n
    cnt = np.array([r["n_success"] for r in all_rows])
    p_base = float(cnt.sum() / (len(cnt) * N)) if len(cnt) else 0.0
    oracle = float((cnt > 0).mean()) if len(cnt) else 0.0
    indep = 1 - (1 - p_base) ** N
    always = float((cnt == N).mean()) if len(cnt) else 0.0
    never = float((cnt == 0).mean()) if len(cnt) else 0.0
    # 상태 간 분산 vs 이항분포 분산 (묶임 정도)
    var_obs = float(cnt.var(ddof=0)) if len(cnt) else 0.0
    var_bin = N * p_base * (1 - p_base)
    summary = {"states": len(cnt), "N": N, "p_base": p_base, "oracle": oracle,
               "indep_pred": indep, "never": never, "always": always,
               "var_obs": var_obs, "var_binomial": var_bin,
               "overdispersion": var_obs / var_bin if var_bin > 0 else None,
               "hist": {str(i): int((cnt == i).sum()) for i in range(N + 1)}}
    json.dump({"args": vars(args), "summary": summary, "rows": all_rows},
              open(Path(args.out) / "oracle.json", "w"), ensure_ascii=False, indent=2)
    print("\n== 오라클 Best-of-N")
    print(f"상태 {len(cnt)}개 × 후보 {N}개")
    print(f"기본 성공률 p      {100 * p_base:.0f}%")
    print(f"오라클 (하나라도)  {100 * oracle:.0f}%   ← 후보 선택의 상한")
    print(f"독립 가정 예측     {100 * indep:.0f}%   (성공이 순전히 노이즈 운이라면)")
    print(f"전부 실패한 상태   {100 * never:.0f}%,  전부 성공 {100 * always:.0f}%")
    print(f"과분산 {summary['overdispersion']}  (1 이면 노이즈 운, 크면 상태가 결정)")
    print("성공 개수 분포", summary["hist"])


if __name__ == "__main__":
    main()
