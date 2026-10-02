#!/usr/bin/env python
"""
DAgger — 학습한 모델이 **실제로 가는 상태**에서 전문가가 넘겨받아 정답 동작을 보여 준다 (2026-10-01)

  모델을 평가 때처럼 돌린다 (단독 태스크, 또는 A → 쥔 지 3스텝째 전환 → B).
  그 구간 안 아무 시점(10~200스텝 중 무작위)에서 전문가가 넘겨받아 끝까지 한다.
  넘겨받은 순간부터의 (관측, 전문가 동작)을 그 지시문으로 저장한다.
  모델이 그 전에 이미 성공했으면 저장하지 않는다 (가르칠 것이 없다).

  넘겨받을 때 무엇을 쥐고 있으면: 그 물체가 지금 일에 필요 없으면 바로 내려놓고 시작한다.

⛔ 넘겨받은 뒤 손이 처음 자세 7cm 안으로 들어간 시범은 버린다.
평가용 장면(20~49)은 쓰지 않는다.

쓰는 법
  python dagger_v6.py --policy outputs/v6_model/merged --tasks 0 3 5 --pairs 8:3,8:5 --episodes 0-19,50-69 --out data/v6_dagger1
  실제 로봇 조건(추론 지연 0.56초)에서 모델이 가는 상태를 모으려면 --latency-steps 11
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import collect_expert as ce
import scripted_expert as se
import switch_experiment as sx
import switch_v6 as sv
import task_experts as te
from collect_v6 import ep_range, settle

OBJ_OF_TASK = {1: "akita_black_bowl_1", 3: "akita_black_bowl_1", 4: "akita_black_bowl_1", 8: "akita_black_bowl_1",
               2: "wine_bottle_1", 9: "wine_bottle_1", 6: "cream_cheese_1"}


def held_object(ep):
    """지금 쥐고 있는 물체 (그리퍼가 물체와 닿아 있고 닫는 중). 없으면 None."""
    m, d = ep.inner.sim.model, ep.inner.sim.data
    names = set()
    for c in range(d.ncon):
        cc = d.contact[c]
        a = m.geom_id2name(cc.geom1) or ""
        b = m.geom_id2name(cc.geom2) or ""
        if ("finger" in a) != ("finger" in b):
            other = b if "finger" in a else a
            for obj in ("akita_black_bowl_1", "wine_bottle_1", "cream_cheese_1", "plate_1"):
                if other.startswith(obj):
                    names.add(obj)
    q = float(ep.obs["robot_state"]["gripper"]["qpos"][0])
    return next(iter(names)) if names and q < 0.035 else None


def run_policy_for(ep, runner, chk, k):
    """모델을 최대 k 스텝 돌린다. 그 사이 성공하면 True."""
    lang = chk.language
    for _ in range(k):
        ep.step(ep.policy_action(lang), "B")
        if chk(ep.env):
            return True
    return False


def takeover(ep, task, rec, z_rest):
    """전문가가 지금 상태에서 task 를 끝까지 한다. 필요 없는 물체를 쥐고 있으면 먼저 내려놓는다."""
    held = held_object(ep)
    if held is not None and held != OBJ_OF_TASK.get(task):
        sv.put_down_any(ep, held, z_rest.get(held, ep.obj_pos(held)[2]), rec)
    te.EXPERT[task](ep, rec)


def one(runner, task_a, task_b, i, out, rng, stats, max_policy=200):
    ep = sx.Episode(runner, i)
    runner.policy.reset()
    z_rest = {o: ep.obj_pos(o)[2] for o in ("akita_black_bowl_1", "wine_bottle_1", "cream_cheese_1")}
    chk_a = runner.chk_a
    if task_b is None:                                   # 단독
        task, chk, tag = task_a, chk_a, f"DN{task_a}"
    else:                                                # 전환: 모델로 A 를 하다가 쥔 지 3스텝째에 B
        if not sx_to_switch(ep, runner, task_a):
            ep.env.close()
            return
        task, chk, tag = task_b, runner.chk_b, f"DS{task_a}{task_b}"
        ep.plan, ep.exec_left, ep.plan_norm, ep.pending = np.zeros((0, 7)), 0, None, None
        runner.policy.reset()
    k = int(rng.integers(10, max_policy))
    if run_policy_for(ep, runner, chk, k):               # 이미 성공 — 가르칠 것 없음
        stats["already"] += 1
        ep.env.close()
        return
    t0 = len(ep.log["pos"])
    rec = ce.Rec()
    takeover(ep, task, rec, z_rest)
    ok = settle(ep, chk) and se.min_home_dist(ep, t0) >= 0.07
    stats["tried"] += 1
    if ok and len(rec) > 5:
        rec.save(out / f"{tag}_ep{i}_k{k}.npz", chk.language, source="dagger", takeover_step=k)
        stats["saved"] += 1
        stats["frames"] += len(rec)
    ep.env.close()


def sx_to_switch(ep, runner, a_task):
    import test_redirect as tr
    return tr.to_switch(ep, runner, a_task)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--tasks", type=int, nargs="*", default=[])
    p.add_argument("--pairs", default="")
    p.add_argument("--episodes", default="0-19,50-69")
    p.add_argument("--repeat", type=int, default=1, help="장면마다 넘겨받는 시점을 바꿔 몇 번")
    p.add_argument("--dart", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--latency-steps", type=int, default=0, help="실제 로봇 조건: 추론 지연 (1080 Ti 11스텝)")
    p.add_argument("--max-policy", type=int, default=200)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    se.DART_SIGMA = a.dart
    rng = np.random.default_rng(a.seed)
    out = Path(a.out) / "episodes"
    out.mkdir(parents=True, exist_ok=True)
    stats = {"tried": 0, "saved": 0, "already": 0, "frames": 0}
    jobs = [(t, None) for t in a.tasks] + [tuple(map(int, s.split(":"))) for s in a.pairs.split(",") if s]
    for ta, tb in jobs:
        ra = sx.default_args()
        ra.policy, ra.task_a, ra.task_b = a.policy, ta, tb
        ra.strategy = "flush" if tb is not None else "none"
        ra.latency_steps = a.latency_steps
        runner = sx.Runner(ra)
        s0 = dict(stats)
        for i in ep_range(a.episodes):
            for _ in range(a.repeat):
                torch.manual_seed(int(rng.integers(1 << 30)))
                one(runner, ta, tb, i, out, rng, stats, a.max_policy)
        name = f"T{ta}" if tb is None else f"A{ta}→B{tb}"
        print(f"  == {name}: 저장 {stats['saved'] - s0['saved']}, 모델이 이미 성공 {stats['already'] - s0['already']}",
              flush=True)
        del runner
        torch.cuda.empty_cache()
    print("STATS", json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
