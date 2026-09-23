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
  chain  : 태스크를 연달아 시킨다. 하나가 끝나면 **리셋 없이 그 자리에서** 다음 지시를 주고,
           성공한 구간을 저장한다 (실패하면 거기서 중단). "다른 일을 마친 자리에서 이어서 하기" 데이터로,
           A 재개(B 를 끝낸 자리에서 A 로 돌아가기)와 같은 상태 분포를 만든다.
  strategy : 전환 순간에 지정한 전략(--collect-strategy)을 적용하고, 그 **스크립트 구간까지 포함해** 저장한다.
           "스크립트로 만든 좋은 동작"을 정책에 증류하기 위한 모드. 제약을 지키는 전략에만 쓸 것
           (ret_part f<1, rollback, finish_a). retreat 은 초기 자세 복귀라 쓰면 안 된다.
  rollback : switch 와 같지만, 전환 순간에 **최근 동작을 거꾸로 재생**해 물체를 집었던 자리에 놓은 뒤 B 로 간다.
           되돌리는 구간(R)까지 **학습 데이터에 포함**한다 — 정책이 그 동작을 스스로 하게 만드는 것이 목적(증류).
           초기 자세로 복귀하는 retreat 과 달리 제약을 지키므로 배워도 된다.
  noise  : DART (Laskey 외, CoRL 2017) 방식. 정책이 움직이는 동안 **실행하는 동작에만** 노이즈를 섞어
           궤도를 벗어나게 하고, 저장하는 정답은 **노이즈 없는 정책 동작**으로 한다. 성공한 에피소드만 저장.
           → "벗어난 상태에서 제자리로 돌아오는" 교정 동작 데이터. 노이즈는 구간(burst)으로 넣어 실제로 밀려나게 함

둘 다 자기가 성공한 궤적을 다시 배우는 self-imitation 이라 사람 시연이 필요 없다.

저장: 에피소드당 npz 1개 — 이미지 2장(JPEG), 끝단 위치·자세, 그리퍼, 실행한 동작

예시:
  python collect_robust_data.py --tasks 0 1 2 4 5 7 8 --episodes 20 --max-offset-cm 8 --max-yaw-deg 20
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

import switch_experiment as sx


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="HuggingFaceVLA/smolvla_libero")
    p.add_argument("--mode",
                   choices=["pose", "switch", "hindsight", "chain", "noise", "rollback",
                            "strategy", "dropped"],
                   default="pose")
    p.add_argument("--obj-displace-cm", type=float, default=8.0,
                   help="dropped 모드: 쥔 물체를 원래 자리에서 이만큼 옮겨 놓고 시작한다. "
                        "작게 시작해 넓혀 가는 **역커리큘럼**에 쓴다 (한 번에 크게 주면 데이터가 안 모인다)")
    p.add_argument("--collect-strategy", default="ret_part",
                   help="strategy 모드: 전환 순간에 적용할 전략 (ret_part / rollback / finish_a / flush 등). "
                        "그 전략이 만드는 **스크립트 구간까지 학습 데이터에 포함**된다")
    p.add_argument("--collect-retreat-frac", type=float, default=0.5,
                   help="strategy 모드에서 ret_part 를 쓸 때의 되돌림 비율")
    p.add_argument("--noise-std", type=float, default=0.4, help="noise 모드: 위치 동작 노이즈 표준편차 (동작 단위, 1.0=5cm)")
    p.add_argument("--noise-burst", type=int, default=6, help="noise 모드: 한 번 노이즈를 넣을 때 이어지는 스텝 수")
    p.add_argument("--noise-prob", type=float, default=0.03, help="noise 모드: 매 스텝 노이즈 구간을 시작할 확률")
    p.add_argument("--noise-max-bursts", type=int, default=2, help="noise 모드: 에피소드당 최대 노이즈 구간 수")
    p.add_argument("--chain-len", type=int, default=3, help="chain 모드: 한 에피소드에서 이어서 시킬 태스크 수")
    p.add_argument("--rollback-pre", type=int, default=10, help="rollback 모드: 집기 시작보다 몇 스텝 더 앞까지 되돌릴지")
    p.add_argument("--rollback-max", type=int, default=60, help="rollback 모드: 되돌릴 최대 스텝 수")
    p.add_argument("--tasks", type=int, nargs="+", required=True, help="pose: 대상 태스크 / switch: 태스크 A 목록")
    p.add_argument("--task-b", type=int, nargs="+", default=None, help="switch 모드에서 끼어들 태스크 (A 와 짝지음)")
    p.add_argument("--switch-at", default="grasp:3", help="switch 모드 전환 시점")
    p.add_argument("--episodes", type=int, default=20, help="태스크당 시도 횟수")
    p.add_argument("--start-episode", type=int, default=0)
    p.add_argument("--max-offset-cm", type=float, default=8.0)
    p.add_argument("--max-yaw-deg", type=float, default=20.0)
    p.add_argument("--min-offset-cm", type=float, default=0.0)
    p.add_argument("--tries", type=int, default=1,
                   help="한 에피소드에서 **성공할 때까지 다시 뽑는** 횟수. "
                        "성공률이 낮은 교란 범위에서는 이게 없으면 데이터가 하나도 안 모인다 "
                        "(2026-09-23: 전환 자세 27cm 에서 물체 집기가 거의 0%). "
                        "같은 시작 자세에 **노이즈만 바꿔** 다시 굴린다")
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




def collect_dropped(a, runner, task, out, stats):
    """⭐ **물체가 낯선 자리에 놓여 있을 때** 그 태스크를 해내는 데이터를 모은다.

    왜 필요한가 (2026-09-23 측정)
      전환하면 로봇이 물체를 놓는데, **놓인 자리는 학습에서 본 적 없는 곳**이다:
          정상 그릇 위치의 퍼짐  ±1.4cm
          놓인 뒤 위치          정상에서 8.3cm 떨어짐 = **6 표준편차 밖**
      그런데 `pose` 모드는 **팔만 교란하고 물체는 제자리**에 둔다.
      실제 전환은 **팔도 교란 + 물체도 낯선 자리**다 → 그대로 학습하면 빈틈이 남는다.
      (LoRA 2차가 '학습한 상황과 실제 상황이 다른' 같은 실수로 실패했다)

    어떻게 하나
      ① 태스크를 하다 물체를 쥐면
      ② **그 물체를 지정한 거리만큼 옮겨 놓고 그리퍼를 연다** (놓기 — 정책이 어차피 하는 동작)
      ③ 그 자리에서 **같은 태스크를 다시** 시킨다 — 물체가 낯선 자리에 있는 상태로
      ④ 성공하면 **놓은 뒤 구간만** 저장한다 (옮기는 구간은 스크립트라 저장하지 않는다)

    ⚠️ 옮기는 거리를 크게 주면 성공률이 0 이 되어 데이터가 안 모인다.
       작게 시작해 넓혀 가야 한다 (`--obj-displace-cm`).
    """
    chk = sx.GoalChecker(runner.suite, task)
    for i in range(a.start_episode, a.start_episode + a.episodes):
        stats["tried"] += 1
        ep = sx.Episode(runner, i)
        runner.policy.reset()

        # ① 물체를 쥘 때까지
        grasp = {"t": None}

        def watch():
            if grasp["t"] is None and ep.holding():
                grasp["t"] = len(ep.log["pos"])

        def trigger(t):
            return grasp["t"] is not None and t >= grasp["t"] + 3

        why, _ = ep.run_policy(chk.language, "A", a.max_steps, chk, trigger, watch)
        if why != "trigger" or not sx.gripper_closed(ep.obs):
            ep.env.close()
            continue
        stats["reached"] += 1

        # ② 지정 거리만큼 옮기고 놓는다 (이 구간은 저장하지 않는다)
        rng = np.random.default_rng(3000 + 17 * i + task)
        d = rng.normal(size=3)
        d[2] = abs(d[2]) * 0.3          # 위로 던지지 않게
        d /= np.linalg.norm(d)
        tgt = sx.eef_pos(ep.obs) + d * (a.obj_displace_cm / 100.0)
        move_to(ep, tgt, ep.obs["robot_state"]["eef"]["mat"].copy())
        for _ in range(15):             # 그리퍼를 열어 놓는다
            ep.step(np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32), "R")
        for _ in range(10):             # 물체가 안정될 시간
            ep.step(sx.get_libero_dummy_action(), "R")

        # ③ 그 자리에서 같은 태스크를 다시 — 이제부터가 학습 데이터다
        frames = {"img": [], "wrist": [], "eef_pos": [], "eef_quat": [], "grip": [], "action": []}
        state0 = ep.env._env.get_sim_state().copy() if a.tries > 1 else None
        success, n_step, used = False, 0, 0
        for attempt in range(max(a.tries, 1)):
            used = attempt + 1
            if attempt > 0:
                ep.inner.timestep = 0
                ep.inner.done = False
                raw = ep.env._env.regenerate_obs_from_state(state0)
                ep.obs = ep.env._format_raw_obs(raw)
                ep.plan, ep.exec_left, ep.plan_norm, ep.pending = np.zeros((0, 7)), 0, None, None
                runner.policy.reset()
                torch.manual_seed(20_000 + 97 * attempt + i)
            frames = {k: [] for k in frames}
            success, n_step = False, 0
            for s in range(a.max_steps):
                o = ep.obs
                frames["img"].append(jpeg(o["pixels"]["image"], a.jpeg_quality))
                frames["wrist"].append(jpeg(o["pixels"]["image2"], a.jpeg_quality))
                frames["eef_pos"].append(np.asarray(o["robot_state"]["eef"]["pos"], np.float32))
                frames["eef_quat"].append(np.asarray(o["robot_state"]["eef"]["quat"], np.float32))
                frames["grip"].append(np.asarray(o["robot_state"]["gripper"]["qpos"], np.float32))
                action = ep.policy_action(chk.language)
                frames["action"].append(np.asarray(action, np.float32))
                ep.step(action, "A")
                n_step = s + 1
                if chk(ep.env):
                    success = True
                    break
            if success:
                break

        if success:
            np.savez(out / "episodes" / f"D{task}_ep{i}.npz",
                     img=np.array(frames["img"], dtype=object),
                     wrist=np.array(frames["wrist"], dtype=object),
                     eef_pos=np.stack(frames["eef_pos"]), eef_quat=np.stack(frames["eef_quat"]),
                     grip=np.stack(frames["grip"]), action=np.stack(frames["action"]),
                     task=chk.language, obj_displace_cm=a.obj_displace_cm)
            stats["saved"] += 1
            stats["frames"] += n_step
        print(json.dumps({"mode": "dropped", "task": task, "episode": i,
                          "obj_displace_cm": a.obj_displace_cm,
                          "success": success, "steps": n_step, "tries_used": used}), flush=True)
        ep.env.close()


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

        # rollback 모드: 되돌릴 동작을 미리 만들어 둔다. 기록 루프 안에서 실행해야 R 구간도 저장된다.
        rb = []
        if a.mode == "rollback" and ep.grasp_t is not None and sx.gripper_closed(ep.obs):
            hi, g = len(ep.acts), ep.grasp_t
            lo = max(0, g - a.rollback_pre, hi - a.rollback_max)
            opened = False
            for k in range(hi - 1, lo - 1, -1):
                if k <= g:
                    opened = True  # 집었던 시점을 지나면 그리퍼를 연다 = 물체를 그 자리에 놓음
                rb.append(np.concatenate([-ep.acts[k][:6], [-1.0 if opened else 1.0]]).astype(np.float32))
            rb += [np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32)] * 10  # 그리퍼가 열릴 시간

        # hindsight: 전환 이후 "우연히 달성된" 다른 태스크도 찾기 위해 10개 goal 을 모두 감시
        others = {}
        if a.mode == "hindsight":
            for tid in range(10):
                c = sx.GoalChecker(runner.suite, tid)
                if not c(ep.env):  # 전환 시점에 이미 만족된 것은 제외
                    others[tid] = c

        frames = {"img": [], "wrist": [], "eef_pos": [], "eef_quat": [], "grip": [], "action": []}

        def push(o, action):
            frames["img"].append(jpeg(o["pixels"]["image"], a.jpeg_quality))
            frames["wrist"].append(jpeg(o["pixels"]["image2"], a.jpeg_quality))
            frames["eef_pos"].append(np.asarray(o["robot_state"]["eef"]["pos"], np.float32))
            frames["eef_quat"].append(np.asarray(o["robot_state"]["eef"]["quat"], np.float32))
            frames["grip"].append(np.asarray(o["robot_state"]["gripper"]["qpos"], np.float32))
            frames["action"].append(np.asarray(action, np.float32))

        if a.mode == "strategy":
            # 전환 처리(스크립트 구간 포함)를 기록 훅으로 잡는다
            ep.record = lambda o, act, ph: push(o, act)
            ep.apply_strategy(chk_b.language, {}, "to_b")
            ep.record = None

        achieved = {}  # task_id -> (지시문, 달성된 스텝)
        success, n = False, 0
        for t in range(a.max_steps):
            o = ep.obs
            action = rb.pop(0) if rb else ep.policy_action(chk_b.language)
            push(o, action)
            ep.step(action, "R" if rb else "B")
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
            pre = {"rollback": "R", "strategy": "G"}.get(a.mode, "S")
            save(f"{pre}{task_a}to{task_b}_ep{i}", chk_b.language, len(frames["action"]), a.mode)
        for tid, (lang, upto) in achieved.items():
            if success and lang == chk_b.language:
                continue  # 위에서 이미 저장
            save(f"H{task_a}to{tid}_ep{i}", lang, upto, "hindsight")
        print(json.dumps({"mode": a.mode, "task_a": task_a, "task_b": task_b, "episode": i,
                          "b_success": success, "steps": n,
                          "hindsight": [v[0] for v in achieved.values()]}, ensure_ascii=False), flush=True)
        ep.env.close()


# 서로 목표가 충돌하지 않는 조합 (물체를 집는 태스크 1개 + 고정물 태스크들)
PICK_TASKS = [1, 2, 4, 8]      # bowl→stove, bottle→cabinet top, bowl→cabinet top, bowl→plate
FIXTURE_TASKS = [0, 5, 7]      # open middle drawer, push plate, turn on stove
CONFLICT = {8: {5}}            # 그릇을 접시에 놓은 뒤 접시를 밀면 앞 태스크가 깨짐


def collect_chain(a, rng, out, stats):
    """태스크를 리셋 없이 연달아 수행. 두 번째 태스크부터의 성공 구간을 저장."""
    for i in range(a.start_episode, a.start_episode + a.episodes):
        pick = int(rng.choice(PICK_TASKS))
        fixtures = [f for f in FIXTURE_TASKS if f not in CONFLICT.get(pick, set())]
        seq = [pick] + list(rng.permutation(fixtures))
        rng.shuffle(seq)
        seq = [int(x) for x in seq[: a.chain_len]]

        rargs = SimpleNamespace(policy=a.policy, task_a=seq[0], task_b=None, switch_at="step:99999",
                                strategy="flush", blend_steps=10, n_action_steps=a.n_action_steps,
                                max_steps=a.max_steps, episodes=1, start_episode=i, video=False, out=a.out,
                                seed=a.seed)
        if not hasattr(collect_chain, "runner") or collect_chain.runner.args.task_a != seq[0]:
            collect_chain.runner = sx.Runner(rargs)
        runner = collect_chain.runner
        runner.args = sx.fill_defaults(rargs)
        ep = sx.Episode(runner, i)
        runner.policy.reset()
        checkers = {tid: sx.GoalChecker(runner.suite, tid) for tid in seq}
        log = {"mode": "chain", "episode": i, "sequence": seq, "done": []}

        for k, tid in enumerate(seq):
            chk = checkers[tid]
            if chk(ep.env):
                log["done"].append(f"{tid}:이미 만족")
                continue
            stats["tried"] += 1
            ep.plan, ep.exec_left = np.zeros((0, 7)), 0  # 그 자리에서 새 지시 (리셋 없음)
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
                ep.step(action, "A" if k == 0 else "B")
                n = t + 1
                if chk(ep.env):
                    success = True
                    break
            # 앞 태스크를 망가뜨렸는지 (예: 그릇을 다시 치움)
            broken = [p for p in seq[:k] if not checkers[p](ep.env)]
            log["done"].append(f"{tid}:{'성공' if success else '실패'}({n})" + (f" 앞태스크 깨짐 {broken}" if broken else ""))
            if not success:
                break
            if k > 0:  # 첫 태스크는 초기 자세에서 시작하는 일반 데이터라 제외
                stats["reached"] += 1
                np.savez(out / "episodes" / f"C{i}_{k}_T{tid}.npz",
                         img=np.array(frames["img"], dtype=object), wrist=np.array(frames["wrist"], dtype=object),
                         eef_pos=np.stack(frames["eef_pos"]), eef_quat=np.stack(frames["eef_quat"]),
                         grip=np.stack(frames["grip"]), action=np.stack(frames["action"]),
                         task=chk.language, mode="chain", previous=str(seq[:k]), broken=str(broken))
                stats["saved"] += 1
                stats["frames"] += n
        print(json.dumps(log, ensure_ascii=False), flush=True)
        ep.env.close()


def collect_noise(a, rng, runner, task, out, stats):
    """DART 방식: 실행 동작에만 노이즈, 저장 정답은 깨끗한 정책 동작. 성공 에피소드만 저장."""
    chk = runner.chk_a
    for i in range(a.start_episode, a.start_episode + a.episodes):
        stats["tried"] += 1
        stats["reached"] += 1
        ep = sx.Episode(runner, i)
        runner.policy.reset()
        frames = {"img": [], "wrist": [], "eef_pos": [], "eef_quat": [], "grip": [], "action": []}
        success, n, burst_left, noisy_steps, bursts = False, 0, 0, 0, 0
        burst = np.zeros(7)
        for t in range(a.max_steps):
            o = ep.obs
            frames["img"].append(jpeg(o["pixels"]["image"], a.jpeg_quality))
            frames["wrist"].append(jpeg(o["pixels"]["image2"], a.jpeg_quality))
            frames["eef_pos"].append(np.asarray(o["robot_state"]["eef"]["pos"], np.float32))
            frames["eef_quat"].append(np.asarray(o["robot_state"]["eef"]["quat"], np.float32))
            frames["grip"].append(np.asarray(o["robot_state"]["gripper"]["qpos"], np.float32))
            clean = np.asarray(ep.policy_action(chk.language), np.float32)
            frames["action"].append(clean)  # 정답 = 노이즈 없는 동작
            if burst_left == 0 and bursts < a.noise_max_bursts and rng.random() < a.noise_prob:
                burst_left = a.noise_burst
                bursts += 1
                burst = np.zeros(7)
                burst[:3] = rng.normal(0, a.noise_std, 3)  # 위치만, 구간 동안 같은 방향으로 밀기
            executed = clean.copy()
            if burst_left > 0:
                executed[:3] = np.clip(executed[:3] + burst[:3], -1, 1)
                burst_left -= 1
                noisy_steps += 1
            ep.step(executed, "A")
            n = t + 1
            if chk(ep.env):
                success = True
                break
        if success and noisy_steps > 0:
            np.savez(out / "episodes" / f"N{task}_ep{i}.npz",
                     img=np.array(frames["img"], dtype=object), wrist=np.array(frames["wrist"], dtype=object),
                     eef_pos=np.stack(frames["eef_pos"]), eef_quat=np.stack(frames["eef_quat"]),
                     grip=np.stack(frames["grip"]), action=np.stack(frames["action"]),
                     task=chk.language, mode="noise", noisy_steps=noisy_steps)
            stats["saved"] += 1
            stats["frames"] += n
        print(json.dumps({"mode": "noise", "task": task, "episode": i, "success": success, "steps": n,
                          "noisy_steps": noisy_steps}, ensure_ascii=False), flush=True)
        ep.env.close()


def main():
    a = parse_args()
    out = Path(a.out)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    stats = {"tried": 0, "reached": 0, "saved": 0, "frames": 0}
    t_start = time.time()

    if a.mode == "chain":
        collect_chain(a, rng, out, stats)
        pairs = []
    else:
        pairs = (list(zip(a.tasks, a.task_b))
                 if a.mode in ("switch", "hindsight", "rollback", "strategy")
                 else [(t, None) for t in a.tasks])
    for task, task_b in pairs:
        rargs = SimpleNamespace(policy=a.policy, task_a=task, task_b=task_b,
                                switch_at=(a.switch_at
                                           if a.mode in ("switch", "hindsight", "rollback", "strategy")
                                           else "step:99999"),
                                strategy=(a.collect_strategy if a.mode == "strategy" else "flush"),
                                retreat_frac=a.collect_retreat_frac,
                                blend_steps=10, n_action_steps=a.n_action_steps,
                                max_steps=a.max_steps, episodes=a.episodes, start_episode=a.start_episode,
                                video=False, out=a.out, seed=a.seed)
        runner = sx.Runner(rargs)
        chk = runner.chk_a
        if a.mode in ("switch", "hindsight", "rollback", "strategy"):
            collect_switch(a, runner, task, task_b, out, stats)
            continue
        if a.mode == "noise":
            collect_noise(a, rng, runner, task, out, stats)
            continue
        if a.mode == "dropped":
            collect_dropped(a, runner, task, out, stats)
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

            # 여기서부터 정책이 만든 궤적만 기록.
            # --tries > 1 이면 **같은 시작 자세를 저장해 두고 노이즈만 바꿔** 다시 굴린다.
            # 교란이 클수록 성공률이 낮아 한 번에 성공하지 못하기 때문이다.
            # 상태 저장·복원은 oracle_bon.py 에서 검증된 방식을 그대로 쓴다
            state0 = ep.env._env.get_sim_state().copy() if a.tries > 1 else None
            success, n, used = False, 0, 0
            frames = None
            for attempt in range(max(a.tries, 1)):
                used = attempt + 1
                if attempt > 0:
                    # 같은 시작 자세로 되돌리고 **노이즈만** 바꾼다.
                    # 로봇수트는 스텝 카운터가 horizon 을 넘으면 step 을 거부하므로 함께 되돌린다.
                    ep.inner.timestep = 0
                    ep.inner.done = False
                    raw = ep.env._env.regenerate_obs_from_state(state0)
                    ep.obs = ep.env._format_raw_obs(raw)
                    ep.plan, ep.exec_left, ep.plan_norm, ep.pending = np.zeros((0, 7)), 0, None, None
                    ep.last_grip = float(ep.obs["robot_state"]["gripper"]["qpos"][0] > 0.02) * 2 - 1
                    runner.policy.reset()
                    torch.manual_seed(10_000 + 97 * attempt + i)
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
                              "yaw_deg": round(yaw, 1), "success": success, "steps": n,
                              "tries_used": used}), flush=True)

    stats["wall_sec"] = round(time.time() - t_start, 1)
    stats["success_rate"] = round(stats["saved"] / max(stats["reached"], 1), 3)
    (out / "collect_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print("STATS", json.dumps(stats, ensure_ascii=False), flush=True)
    # 안전장치: 한 개도 저장하지 못했으면 실패로 끝낸다.
    # (2026-09-18 교훈 — 옵션 누락 버그로 수집이 전부 죽었는데도 파이프라인이 그대로 학습까지 진행했다)
    if stats["saved"] == 0:
        print("ERROR 저장된 에피소드가 0개입니다 — 수집이 실패했습니다.", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
