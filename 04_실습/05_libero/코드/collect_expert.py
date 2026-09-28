#!/usr/bin/env python
"""
전문가 시범 수집 — LoRA 5차용

── LoRA 4차가 왜 실패했나 (2026-09-28 분석) ──────────────────────────────
  ① **정책 자신의 성공을 따라하게** 했다 (self-imitation).
     그런데 정책은 교란된 자세에서 거의 성공하지 못한다 → 다시 뽑기로 억지로 모은 성공은
     **운 좋은 노이즈 덕**이라, 따라하면 오히려 망가진다.
     → 학습한 태스크 T8 이 교란 없는 상태에서도 90% → 10%
  ② **태스크 3개에만 몰렸다** (158개 에피소드). 2차는 442개로 10개 태스크를 모두 덮었다
  ③ 검증 손실이 **거의 안 내려갔다** (0.1295 → 0.1286) — 데이터가 배우기 어려운 종류였다

── 5차는 이렇게 다르다 ────────────────────────────────────────────────────
  ① **스크립트 전문가**가 시범을 보인다. 시뮬레이터가 물체 위치를 알아 어디서든 해낸다
  ② **정책이 실제로 가는 상태**(전환 순간)에서 시범을 보인다  📚 DAgger (2011)
  ③ **10개 태스크 전부**를 리허설로 섞는다 — 망각 방지

── 무엇을 모으나 ──────────────────────────────────────────────────────────
  mode=putdown   전환 순간 → 전문가가 그릇을 **바로 세워** 내려놓는다 → 정책이 B 를 한다.
                 전체(내려놓기 + B)를 B 의 지시문으로 저장. 정책은 "내려놓기"를 배운다.
  mode=redirect  전환 순간 → 전문가가 그릇을 **쥔 채** 새 목적지로 가져간다
  mode=normal    교란 없이 정책이 스스로 성공한 것 (리허설, 10개 태스크 전부)

⚠️ 제약: 전문가는 초기 자세로 돌아가지 않는다. 물체를 내려놓는 것은 허용 범위('아무 데나').
"""
import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import scripted_expert as se
import switch_experiment as sx
import test_redirect as tr


def jpeg(img, q=90):
    b = io.BytesIO()
    Image.fromarray(img).save(b, format="JPEG", quality=q)
    return b.getvalue()


class Rec:
    """(관측, 동작) 을 모으는 기록기. 전문가 구간과 정책 구간 모두 같은 형식으로."""

    def __init__(self):
        self.f = {"img": [], "wrist": [], "eef_pos": [], "eef_quat": [], "grip": [], "action": []}

    def __call__(self, o, a):
        self.f["img"].append(jpeg(o["pixels"]["image"]))
        self.f["wrist"].append(jpeg(o["pixels"]["image2"]))
        self.f["eef_pos"].append(np.asarray(o["robot_state"]["eef"]["pos"], np.float32))
        self.f["eef_quat"].append(np.asarray(o["robot_state"]["eef"]["quat"], np.float32))
        self.f["grip"].append(np.asarray(o["robot_state"]["gripper"]["qpos"], np.float32))
        self.f["action"].append(np.asarray(a, np.float32))

    def __len__(self):
        return len(self.f["action"])

    def save(self, path, task, **meta):
        np.savez(path, img=np.array(self.f["img"], dtype=object),
                 wrist=np.array(self.f["wrist"], dtype=object),
                 eef_pos=np.stack(self.f["eef_pos"]), eef_quat=np.stack(self.f["eef_quat"]),
                 grip=np.stack(self.f["grip"]), action=np.stack(self.f["action"]),
                 task=task, **meta)


def run_policy_rec(ep, rec, instruction, chk, max_steps):
    """정책을 돌리며 기록한다. 성공하면 True."""
    for _ in range(max_steps):
        a = ep.policy_action(instruction)
        rec(ep.obs, a)
        ep.step(a, "B")
        if chk(ep.env):
            return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["putdown", "redirect", "normal"], required=True)
    p.add_argument("--pairs", default="8:5,8:7,8:0,4:5,4:7,4:0,1:5,1:0")
    p.add_argument("--tasks", type=int, nargs="+", default=list(range(10)), help="normal 모드")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--start-episode", type=int, default=0)
    p.add_argument("--tries", type=int, default=3, help="정책 구간(B)을 다시 뽑는 횟수")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    out = Path(a.out) / "episodes"
    out.mkdir(parents=True, exist_ok=True)
    stats = {"tried": 0, "saved": 0, "frames": 0, "per": {}}
    t0 = time.time()

    if a.mode == "normal":
        for task in a.tasks:
            ra = sx.default_args()
            ra.task_a, ra.strategy = task, "none"
            runner = sx.Runner(ra)
            chk = runner.chk_a
            k = 0
            for i in range(a.start_episode, a.start_episode + a.episodes):
                stats["tried"] += 1
                ep = sx.Episode(runner, i)
                runner.policy.reset()
                rec = Rec()
                ok = run_policy_rec(ep, rec, chk.language, chk, a.max_steps)
                if ok:
                    rec.save(out / f"N{task}_ep{i}.npz", chk.language, source="normal")
                    stats["saved"] += 1
                    stats["frames"] += len(rec)
                    k += 1
                ep.env.close()
            stats["per"][f"T{task}"] = k
            print(f"T{task}: {k}/{a.episodes} 저장", flush=True)
    else:
        for pr in a.pairs.split(","):
            at, bt = map(int, pr.split(":"))
            ra = sx.default_args()
            ra.task_a, ra.task_b, ra.strategy = at, bt, "flush"
            runner = sx.Runner(ra)
            chk_b = sx.GoalChecker(runner.suite, bt)
            k = 0
            for i in range(a.start_episode, a.start_episode + a.episodes):
                stats["tried"] += 1
                ep = sx.Episode(runner, i)
                runner.policy.reset()
                if not tr.to_switch(ep, runner, at):
                    ep.env.close()
                    continue
                state0 = ep.env._env.get_sim_state().copy()
                saved = False
                for attempt in range(max(a.tries, 1)):
                    if attempt > 0:  # 전환 순간으로 되돌리고 노이즈만 바꿔 다시
                        ep.inner.timestep = 0
                        ep.inner.done = False
                        ep.obs = ep.env._format_raw_obs(ep.env._env.regenerate_obs_from_state(state0))
                        ep.plan, ep.exec_left, ep.plan_norm, ep.pending = np.zeros((0, 7)), 0, None, None
                        torch.manual_seed(30_000 + 97 * attempt + i)
                    rec = Rec()
                    if a.mode == "redirect":
                        ok, _ = se.redirect(ep, bt, record=rec)
                    else:
                        if not se.put_down(ep, record=rec):
                            continue  # 바로 세우지 못한 시범은 쓰지 않는다
                        ep.plan, ep.exec_left, ep.plan_norm = np.zeros((0, 7)), 0, None
                        runner.policy.reset()
                        ok = run_policy_rec(ep, rec, chk_b.language, chk_b, a.max_steps)
                    if ok:
                        rec.save(out / f"{a.mode[0].upper()}{at}{bt}_ep{i}.npz", chk_b.language,
                                 source=a.mode, pair=pr, tries_used=attempt + 1)
                        stats["saved"] += 1
                        stats["frames"] += len(rec)
                        k += 1
                        saved = True
                        break
                print(f"A{at}→B{bt} ep{i}: {'저장' if saved else '실패'}", flush=True)
                ep.env.close()
            stats["per"][pr] = k
            print(f"  == A{at}→B{bt}: {k}/{a.episodes} 저장", flush=True)

    stats["wall_sec"] = round(time.time() - t0, 1)
    (Path(a.out) / "collect_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print("STATS", json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
