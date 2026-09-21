#!/usr/bin/env python
"""
자세 민감도: 시작 자세를 초기 자세에서 얼마나 틀면 SmolVLA 성공률이 무너지는가.

전환 실험에서 "전환 시점의 로봇 자세"가 지배 변수로 보였기 때문에, 전환을 빼고
태스크 하나만으로 그 관계를 직접 잰다.

절차 (에피소드 1회):
  1) 환경 리셋 (LIBERO 고정 초기상태)
  2) 스크립트 P 제어로 끝단을 [초기 자세 + 위치 오프셋 / 손목 yaw 회전] 으로 이동
  3) 그 자세에서 정책에 태스크 지시를 주고 성공 여부 측정

예시:
  python pose_sensitivity.py --task 7 --offset-cm 6 --episodes 10
  python pose_sensitivity.py --task 7 --yaw-deg 20 --episodes 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import switch_experiment as sx


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="HuggingFaceVLA/smolvla_libero")
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--offset-cm", type=float, default=0.0, help="초기 자세에서 끝단을 옮길 거리(cm)")
    p.add_argument("--yaw-deg", type=float, default=0.0, help="손목 회전(월드 z축, 도)")
    p.add_argument("--n-action-steps", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--start-episode", type=int, default=0)
    p.add_argument("--video", action="store_true")
    p.add_argument("--save-traj", action="store_true",
                   help="궤적(위치·손목 회전)을 npz 로 저장. 교란된 손목이 실제로 고쳐지는지 보려면 필요하다")
    p.add_argument("--out", default="outputs/pose")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def unit_dir(rng):
    """무작위 방향 (아래로 파고들지 않도록 z 성분은 양수 쪽으로 제한)."""
    v = rng.normal(size=3)
    v[2] = abs(v[2]) * 0.5
    return v / np.linalg.norm(v)


def move_to(ep, target_pos, target_mat, max_steps=150):
    """P 제어로 목표 끝단 자세까지 이동. (도달 여부, 사용 스텝, 위치 오차) 반환."""
    for _ in range(max_steps):
        pos = sx.eef_pos(ep.obs)
        dp = target_pos - pos
        rot_err = Rotation.from_matrix(target_mat @ ep.obs["robot_state"]["eef"]["mat"].T).as_rotvec()
        if np.linalg.norm(dp) < 0.01 and np.linalg.norm(rot_err) < 0.05:
            break
        ep.step(np.concatenate([np.clip(dp / sx.POS_SCALE, -1, 1),
                                np.clip(rot_err / sx.ROT_SCALE, -1, 1), [-1]]), "R")
    for _ in range(5):  # 안정화
        ep.step(sx.get_libero_dummy_action(), "R")
    err = float(np.linalg.norm(target_pos - sx.eef_pos(ep.obs)))
    return err < 0.02, len(ep.log["pos"]), err


def main():
    a = parse_args()
    rargs = SimpleNamespace(policy=a.policy, task_a=a.task, task_b=None, switch_at="step:999",
                            strategy="none", blend_steps=10, n_action_steps=a.n_action_steps,
                            max_steps=a.max_steps, episodes=a.episodes, start_episode=a.start_episode,
                            video=a.video, out=a.out, seed=a.seed)
    runner = sx.Runner(rargs)
    chk = runner.chk_a
    tag = f"T{a.task}_off{a.offset_cm:g}_yaw{a.yaw_deg:g}"
    Path(a.out).mkdir(parents=True, exist_ok=True)
    recs = []

    for ep_idx in range(a.start_episode, a.start_episode + a.episodes):
        t0 = time.time()
        ep = sx.Episode(runner, ep_idx)
        runner.policy.reset()
        rng = np.random.default_rng(1000 * a.task + ep_idx)  # 방향은 조건과 무관하게 고정
        home_pos, home_mat = ep.home
        direction = unit_dir(rng)
        target_pos = home_pos + direction * (a.offset_cm / 100.0)
        target_mat = Rotation.from_rotvec([0, 0, np.deg2rad(a.yaw_deg)]).as_matrix() @ home_mat
        reached, move_steps, err = move_to(ep, target_pos, target_mat)
        why, n = ep.run_policy(chk.language, "A", a.max_steps, chk)
        rec = {"episode": ep_idx, "task": a.task, "instruction": chk.language,
               "offset_cm": a.offset_cm, "yaw_deg": a.yaw_deg,
               "direction": [round(float(x), 3) for x in direction],
               "reached_start_pose": bool(reached), "start_pose_err_cm": round(err * 100, 2),
               "move_steps": move_steps, "success": why == "success", "steps": n,
               "wall_sec": round(time.time() - t0, 1)}
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        recs.append(rec)
        if a.save_traj:
            # 조건마다 이름이 달라야 한다 (Runner.tag 는 조건을 구분하지 못한다)
            td = Path(a.out) / "traj"
            td.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                td / f"{tag}_ep{ep_idx}.npz",
                pos=np.asarray(ep.log["pos"], dtype=np.float32),
                phase=np.asarray(ep.log["phase"]),          # R=스크립트 이동, A=정책
                eef_rotvec=np.asarray(ep.log["rot"], dtype=np.float32),
                home_rotvec=Rotation.from_matrix(home_mat).as_rotvec().astype(np.float32),
                home_pos=np.asarray(home_pos, dtype=np.float32),
                target_rotvec=Rotation.from_matrix(target_mat).as_rotvec().astype(np.float32),
                target_pos=np.asarray(target_pos, dtype=np.float32),
                success=bool(why == "success"))
        runner.finish(ep, {}, time.time(), ep_idx) if a.video else ep.env.close()

    ok = [r for r in recs if r["reached_start_pose"]]
    summary = {"n": len(recs), "n_reached": len(ok),
               "success_rate": round(float(np.mean([r["success"] for r in ok])), 3) if ok else None,
               "steps_median_success": float(np.median([r["steps"] for r in ok if r["success"]] or [np.nan]))}
    print("SUMMARY", tag, json.dumps(summary, ensure_ascii=False), flush=True)
    with open(Path(a.out) / f"{tag}.json", "w") as f:
        json.dump({"args": vars(a), "summary": summary, "episodes": recs}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
