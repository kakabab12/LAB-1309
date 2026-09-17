#!/usr/bin/env python
"""
자세 강건화 학습 데이터 수집.

배경: 전환 실패의 원인이 "전환 시점의 로봇 자세가 학습 분포를 벗어나서"라는 것을 확인했다
      (2026-09-16 자세 민감도). 그런데 전환할 때 초기 자세로 되돌리는 건 금지 (리셋이 되어버림).
      → 정책 자체가 어떤 자세에서든 작동하게 만들어야 한다.

방법 (--mode)
  pose   : 시작 자세를 무작위로 틀어놓고 정책을 돌린 뒤 **성공한 에피소드만** 저장.
           스크립트로 자세를 만드는 구간은 저장하지 않는다 (그 동작을 배우면 안 되므로).
  switch : 태스크 A 를 하다 물체를 쥔 상태에서 B 로 지시를 바꾸고(리셋 없음, flush),
           **B 를 성공한 경우만** 전환 시점 이후 구간을 저장. 실제 전환 상황 그대로의 데이터.
  hindsight : switch 와 같지만, B 에 실패해도 버리지 않는다. 전환 이후 구간에서 **우연히 달성된
           다른 태스크의 goal** 을 찾아, 그 태스크의 지시문을 붙여 저장한다 (hindsight relabeling).
           1차 학습에서 성공률이 낮은 쌍은 데이터가 하나도 안 모여 오히려 성능이 떨어졌기 때문에,
           실패에서도 배울 거리를 뽑아내는 방식이 필요하다.

둘 다 자기가 성공한 궤적을 다시 배우는 self-imitation 이라 사람 시연이 필요 없다.

저장: 에피소드당 npz 1개 — 이미지 2장(JPEG), 끝단 위치·자세, 그리퍼, 실행한 동작

예시:
  python collect_robust_data.py --tasks 0 1 2 4 5 7 8 --episodes 20 --max-offset-cm 8 --max-yaw-deg 20
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

import switch_experiment as sx


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="HuggingFaceVLA/smolvla_libero")
    p.add_argument("--mode", choices=["pose", "switch", "hindsight"], default="pose")
    p.add_argument("--tasks", type=int, nargs="+", required=True, help="pose: 대상 태스크 / switch: 태스크 A 목록")
    p.add_argument("--task-b", type=int, nargs="+", default=None, help="switch 모드에서 끼어들 태스크 (A 와 짝지음)")
    p.add_argument("--switch-at", default="grasp:3", help="switch 모드 전환 시점")
    p.add_argument("--episodes", type=int, default=20, help="태스크당 시도 횟수")
    p.add_argument("--start-episode", type=int, default=0)
    p.add_argument("--max-offset-cm", type=float, default=8.0)
    p.add_argument("--max-yaw-deg", type=float, default=20.0)
    p.add_argument("--min-offset-cm", type=float, default=0.0)
    p.add_argument("--n-action-steps", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--out", default="data/robust")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def jpeg(img: np.ndarray, quality: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def move_to(ep, target_pos, target_mat, max_steps=150):
    """P 제어로 목표 끝단 자세까지. (도달 여부, 위치 오차 m) 반환."""
    for _ in range(max_steps):
        dp = target_pos - sx.eef_pos(ep.obs)
        rot_err = Rotation.from_matrix(target_mat @ ep.obs["robot_state"]["eef"]["mat"].T).as_rotvec()
        if np.linalg.norm(dp) < 0.01 and np.linalg.norm(rot_err) < 0.05:
            break
        ep.step(np.concatenate([np.clip(dp / sx.POS_SCALE, -1, 1),
                                np.clip(rot_err / sx.ROT_SCALE, -1, 1), [-1]]), "R")
    for _ in range(5):
        ep.step(sx.get_libero_dummy_action(), "R")
    err = float(np.linalg.norm(target_pos - sx.eef_pos(ep.obs)))
    return err < 0.02, err


def collect_switch(a, runner, task_a, task_b, out, stats):
    """A 를 하다 물체를 쥔 상태에서 B 로 전환(리셋 없음). B 를 성공한 경우 전환 이후 구간만 저장."""
    chk_a, chk_b = runner.chk_a, runner.chk_b
    kind, val = runner.switch_kind, runner.switch_val
    for i in range(a.start_episode, a.start_episode + a.episodes):
        stats["tried"] += 1
        ep = sx.Episode(runner, i)
        runner.policy.reset()
        grasp = {"t": None}

        def watch():
            if grasp["t"] is None and ep.holding():
                grasp["t"] = len(ep.log["pos"])

        def trigger(t):
            return t >= val if kind == "step" else (grasp["t"] is not None and t >= grasp["t"] + val)

        why, _ = ep.run_policy(chk_a.language, "A", a.max_steps, chk_a, trigger, watch)
        if why != "trigger":
            ep.env.close()
            continue
        stats["reached"] += 1
        ep.plan, ep.exec_left = np.zeros((0, 7)), 0  # flush: 남은 chunk 버리고 그 자리에서 전환

        # hindsight: 전환 이후 "우연히 달성된" 다른 태스크도 찾기 위해 10개 goal 을 모두 감시
        others = {}
        if a.mode == "hindsight":
            for tid in range(10):
                c = sx.GoalChecker(runner.suite, tid)
                if not c(ep.env):  # 전환 시점에 이미 만족된 것은 제외
                    others[tid] = c

        frames = {"img": [], "wrist": [], "eef_pos": [], "eef_quat": [], "grip": [], "action": []}
        achieved = {}  # task_id -> (지시문, 달성된 스텝)
        success, n = False, 0
        for t in range(a.max_steps):
            o = ep.obs
            frames["img"].append(jpeg(o["pixels"]["image"], a.jpeg_quality))
            frames["wrist"].append(jpeg(o["pixels"]["image2"], a.jpeg_quality))
            frames["eef_pos"].append(np.asarray(o["robot_state"]["eef"]["pos"], np.float32))
            frames["eef_quat"].append(np.asarray(o["robot_state"]["eef"]["quat"], np.float32))
            frames["grip"].append(np.asarray(o["robot_state"]["gripper"]["qpos"], np.float32))
            action = ep.policy_action(chk_b.language)
            frames["action"].append(np.asarray(action, np.float32))
            ep.step(action, "B")
            n = t + 1
            for tid, c in list(others.items()):
                if c(ep.env):
                    achieved[tid] = (c.language, n)
                    del others[tid]
            if chk_b(ep.env):
                success = True
                break

        def save(tag, instruction, upto, kind):
            np.savez(out / "episodes" / f"{tag}.npz",
                     img=np.array(frames["img"][:upto], dtype=object),
                     wrist=np.array(frames["wrist"][:upto], dtype=object),
                     eef_pos=np.stack(frames["eef_pos"][:upto]), eef_quat=np.stack(frames["eef_quat"][:upto]),
                     grip=np.stack(frames["grip"][:upto]), action=np.stack(frames["action"][:upto]),
                     task=instruction, mode=kind, task_a=chk_a.language, switch_at=a.switch_at)
            stats["saved"] += 1
            stats["frames"] += upto

        if success:
            save(f"S{task_a}to{task_b}_ep{i}", chk_b.language, n, "switch")
        for tid, (lang, upto) in achieved.items():
            if success and lang == chk_b.language:
                continue  # 위에서 이미 저장
            save(f"H{task_a}to{tid}_ep{i}", lang, upto, "hindsight")
        print(json.dumps({"mode": a.mode, "task_a": task_a, "task_b": task_b, "episode": i,
                          "b_success": success, "steps": n,
                          "hindsight": [v[0] for v in achieved.values()]}, ensure_ascii=False), flush=True)
        ep.env.close()


def main():
    a = parse_args()
    out = Path(a.out)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    stats = {"tried": 0, "reached": 0, "saved": 0, "frames": 0}
    t_start = time.time()

    pairs = list(zip(a.tasks, a.task_b)) if a.mode in ("switch", "hindsight") else [(t, None) for t in a.tasks]
    for task, task_b in pairs:
        rargs = SimpleNamespace(policy=a.policy, task_a=task, task_b=task_b,
                                switch_at=a.switch_at if a.mode in ("switch", "hindsight") else "step:99999",
                                strategy="flush", blend_steps=10, n_action_steps=a.n_action_steps,
                                max_steps=a.max_steps, episodes=a.episodes, start_episode=a.start_episode,
                                video=False, out=a.out, seed=a.seed)
        runner = sx.Runner(rargs)
        chk = runner.chk_a
        if a.mode in ("switch", "hindsight"):
            collect_switch(a, runner, task, task_b, out, stats)
            continue

        for i in range(a.start_episode, a.start_episode + a.episodes):
            stats["tried"] += 1
            ep = sx.Episode(runner, i)
            runner.policy.reset()

            # 무작위 시작 자세
            d = rng.normal(size=3)
            d[2] = abs(d[2]) * 0.5
            d /= np.linalg.norm(d)
            off = rng.uniform(a.min_offset_cm, a.max_offset_cm) / 100.0
            yaw = rng.uniform(-a.max_yaw_deg, a.max_yaw_deg)
            home_pos, home_mat = ep.home
            target_mat = Rotation.from_rotvec([0, 0, np.deg2rad(yaw)]).as_matrix() @ home_mat
            reached, err = move_to(ep, home_pos + d * off, target_mat)
            if not reached:
                ep.env.close()
                continue
            stats["reached"] += 1

            # 여기서부터 정책이 만든 궤적만 기록
            frames = {"img": [], "wrist": [], "eef_pos": [], "eef_quat": [], "grip": [], "action": []}
            success, n = False, 0
            for t in range(a.max_steps):
                o = ep.obs
                frames["img"].append(jpeg(o["pixels"]["image"], a.jpeg_quality))
                frames["wrist"].append(jpeg(o["pixels"]["image2"], a.jpeg_quality))
                frames["eef_pos"].append(np.asarray(o["robot_state"]["eef"]["pos"], np.float32))
                frames["eef_quat"].append(np.asarray(o["robot_state"]["eef"]["quat"], np.float32))
                frames["grip"].append(np.asarray(o["robot_state"]["gripper"]["qpos"], np.float32))
                action = ep.policy_action(chk.language)
                frames["action"].append(np.asarray(action, np.float32))
                ep.step(action, "A")
                n = t + 1
                if chk(ep.env):
                    success = True
                    break

            if success:
                np.savez(out / "episodes" / f"T{task}_ep{i}.npz",
                         img=np.array(frames["img"], dtype=object), wrist=np.array(frames["wrist"], dtype=object),
                         eef_pos=np.stack(frames["eef_pos"]), eef_quat=np.stack(frames["eef_quat"]),
                         grip=np.stack(frames["grip"]), action=np.stack(frames["action"]),
                         task=chk.language, offset_cm=off * 100, yaw_deg=yaw, start_pose_err_cm=err * 100)
                stats["saved"] += 1
                stats["frames"] += n
            print(json.dumps({"task": task, "episode": i, "offset_cm": round(off * 100, 1),
                              "yaw_deg": round(yaw, 1), "success": success, "steps": n}), flush=True)

    stats["wall_sec"] = round(time.time() - t_start, 1)
    stats["success_rate"] = round(stats["saved"] / max(stats["reached"], 1), 3)
    (out / "collect_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print("STATS", json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
