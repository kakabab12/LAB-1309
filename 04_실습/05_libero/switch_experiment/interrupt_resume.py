#!/usr/bin/env python
"""
LIBERO_GOAL + SmolVLA: 태스크 인터럽트 & 재개 실험

시나리오 (한 에피소드):
  Phase A  : 태스크 A 지시문으로 수행 → interrupt_step 에서 중단
  Phase B  : 환경 리셋 없이 지시문만 태스크 B 로 교체 → B 완료(또는 시간초과)까지 수행
  Phase A2 : 다시 태스크 A 지시문으로 교체 → A 완료(또는 시간초과)까지 수행

LIBERO_GOAL 은 10개 태스크가 동일한 장면을 공유하므로, 같은 시뮬레이션에서
B 의 BDDL goal predicate 로 B 성공 여부를 판정할 수 있다.

전환 방식 (--switch-mode):
  direct  : 즉시 지시문만 교체 (기존 방식)
  defer   : 물체를 쥐고 있으면 A 를 계속 수행하다가 그리퍼가 열린 뒤(안전 지점)에 전환
  retreat : 스크립트 컨트롤러로 [쥔 물체 내려놓기 → 그리퍼 열기 → 들어올리기 → 초기 자세 복귀]
            후 전환. 학습 데모의 시작 분포(초기 자세, 빈 그리퍼)에 가깝게 만들어 준다.
            A→B, B→A2 두 전환 모두에 적용.

예시:
  python interrupt_resume.py --task-a 8 --task-b 7 --interrupt-step 60 --episodes 5 --video
  python interrupt_resume.py --task-a 8 --task-b 7 --interrupt-step 60 --switch-mode retreat
  python interrupt_resume.py --task-a 8 --interrupt-step -1   # A 단독 베이스라인
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.bddl_utils import robosuite_parse_problem
from scipy.spatial.transform import Rotation

from lerobot.envs.libero import LiberoEnv, get_libero_dummy_action
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.env_processor import LiberoProcessorStep
from lerobot.utils.io_utils import write_video

SUITE = "libero_goal"
GRIPPER_OPEN_QPOS = 0.035  # 손가락 qpos 가 이보다 작으면 닫힘(물체 파지 가능성)으로 간주
POS_SCALE, ROT_SCALE = 0.05, 0.5  # OSC_POSE delta 액션 1.0 당 이동(m) / 회전(rad)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="HuggingFaceVLA/smolvla_libero")
    p.add_argument("--task-a", type=int, required=True, help="LIBERO_GOAL task id (0-9)")
    p.add_argument("--task-b", type=int, default=None, help="인터럽트 태스크 id. 없으면 A 단독")
    p.add_argument("--interrupt-step", type=int, default=60, help="-1 이면 인터럽트 없음(베이스라인)")
    p.add_argument("--max-steps-a", type=int, default=300, help="A 단독/최초 A 단계 최대 스텝")
    p.add_argument("--max-steps-b", type=int, default=300)
    p.add_argument("--max-steps-resume", type=int, default=300)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--start-episode", type=int, default=0, help="LIBERO init state 시작 인덱스")
    p.add_argument("--n-action-steps", type=int, default=None,
                   help="액션 청크에서 몇 스텝 실행 후 재추론할지 (체크포인트 기본=1, 크면 빠름)")
    p.add_argument("--switch-mode", choices=["direct", "defer", "retreat"], default="direct")
    p.add_argument("--defer-max", type=int, default=120, help="defer 모드에서 전환을 미룰 최대 스텝")
    p.add_argument("--no-reset-on-switch", action="store_true",
                   help="전환 시 policy.reset() 을 하지 않음 (남은 액션 큐 유지; ablation 용)")
    p.add_argument("--video", action="store_true")
    p.add_argument("--out", default="outputs/interrupt_resume")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def add_batch_dim(tree):
    if isinstance(tree, dict):
        return {k: add_batch_dim(v) for k, v in tree.items()}
    return np.asarray(tree)[None]


def gripper_closed(obs) -> bool:
    return bool(obs["robot_state"]["gripper"]["qpos"][0] < GRIPPER_OPEN_QPOS)


class GoalChecker:
    """임의의 LIBERO_GOAL 태스크 goal 을 현재 시뮬레이션 상태에서 평가."""

    def __init__(self, suite, task_id: int):
        task = suite.get_task(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        self.goal = robosuite_parse_problem(bddl)["goal_state"]
        self.language = task.language

    def __call__(self, libero_env: LiberoEnv) -> bool:
        inner = libero_env._env.env  # OffScreenRenderEnv -> problem env
        return all(inner._eval_predicate(s) for s in self.goal)


class Runner:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = SmolVLAPolicy.from_pretrained(args.policy)
        self.policy.config.device = self.device
        if args.n_action_steps is not None:
            self.policy.config.n_action_steps = args.n_action_steps
        self.policy.to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=args.policy,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )
        self.env_step = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
        self.suite = benchmark.get_benchmark_dict()[SUITE]()

    @torch.inference_mode()
    def act(self, obs, instruction: str) -> np.ndarray:
        batch = preprocess_observation(add_batch_dim(obs))
        batch["task"] = [instruction]
        batch = self.env_step(batch)
        batch = self.pre(batch)
        action = self.policy.select_action(batch)
        action = self.post(action)
        return action.squeeze(0).cpu().numpy()

    def step(self, env, action, frames, label):
        raw, _, _, _ = env._env.step(np.asarray(action, dtype=np.float64))
        obs = env._format_raw_obs(raw)
        if frames is not None:
            frames.append(self.frame(obs, label))
        return obs

    def run_phase(self, env, obs, instruction, max_steps, success_fn, frames, label, stop_fn=None):
        """success_fn 이 True 가 되거나 (stop_fn 이 True 가 되거나) max_steps 에 도달할 때까지 실행."""
        for t in range(max_steps):
            obs = self.step(env, self.act(obs, instruction), frames, label)
            if success_fn is not None and success_fn(env):
                return obs, True, t + 1
            if stop_fn is not None and stop_fn(obs):
                return obs, False, t + 1
        return obs, False, max_steps

    def retreat(self, env, obs, frames):
        """쥔 물체 내려놓기 → 그리퍼 열기 → 들어올리기 → 초기 자세 복귀. 사용한 스텝 수 반환."""
        home_pos, home_mat = self.home
        n = 0

        def go(action):
            nonlocal obs, n
            obs = self.step(env, action, frames, "R")
            n += 1

        def eef_z():
            return float(obs["robot_state"]["eef"]["pos"][2])

        if gripper_closed(obs):
            # 1) 접촉으로 하강이 멈출 때까지 내림 (물체를 공중에서 떨어뜨리지 않도록)
            zs = []
            for _ in range(40):
                go([0, 0, -0.3, 0, 0, 0, 1])
                zs.append(eef_z())
                if len(zs) >= 5 and zs[-5] - zs[-1] < 0.003:
                    break
        # 2) 그리퍼 열기
        for _ in range(15):
            go([0, 0, 0, 0, 0, 0, -1])
        # 3) 수직으로 들어올리기 (물체와 충돌 방지)
        for _ in range(30):
            if eef_z() >= home_pos[2] - 0.01:
                break
            go([0, 0, 0.6, 0, 0, 0, -1])
        # 4) 초기 자세로 P 제어 (위치 + 월드 좌표계 회전 오차)
        for _ in range(100):
            pos = obs["robot_state"]["eef"]["pos"]
            dp = home_pos - pos
            rot_err = Rotation.from_matrix(home_mat @ obs["robot_state"]["eef"]["mat"].T).as_rotvec()
            if np.linalg.norm(dp) < 0.01 and np.linalg.norm(rot_err) < 0.05:
                break
            a = np.concatenate([np.clip(dp / POS_SCALE, -1, 1), np.clip(rot_err / ROT_SCALE, -1, 1), [-1]])
            go(a)
        # 5) 안정화
        for _ in range(5):
            go(get_libero_dummy_action())
        return obs, n

    @staticmethod
    def frame(obs, label):
        img = obs["pixels"]["image"][::-1, ::-1]
        wrist = obs["pixels"]["image2"][::-1, ::-1]
        f = np.concatenate([img, wrist], axis=1).copy()
        # 단계 표시 색 막대 (A=파랑, B=주황, A2=초록, R=회색: 스크립트 복귀 동작)
        color = {"A": (60, 120, 255), "B": (255, 150, 30), "A2": (40, 200, 90), "R": (160, 160, 160)}[label]
        f[:8] = color
        return f

    def switch(self, env, obs, frames, rec, key):
        steps = 0
        if self.args.switch_mode == "retreat":
            obs, steps = self.retreat(env, obs, frames)
            rec[f"{key}_retreat_steps"] = steps
        if not self.args.no_reset_on_switch:
            self.policy.reset()
        return obs

    def episode(self, ep_idx: int):
        a = self.args
        env = LiberoEnv(
            task_suite=self.suite, task_id=a.task_a, task_suite_name=SUITE,
            obs_type="pixels_agent_pos", episode_index=ep_idx,
        )
        obs, _ = env.reset(seed=a.seed + ep_idx)
        self.home = (obs["robot_state"]["eef"]["pos"].copy(), obs["robot_state"]["eef"]["mat"].copy())
        chk_a = GoalChecker(self.suite, a.task_a)
        chk_b = GoalChecker(self.suite, a.task_b) if a.task_b is not None else None
        frames = [] if a.video else None
        self.policy.reset()
        rec = {"episode": ep_idx, "task_a": chk_a.language,
               "task_b": chk_b.language if chk_b else None}
        t0 = time.time()

        baseline = chk_b is None or a.interrupt_step < 0
        if baseline:
            _, ok, n = self.run_phase(env, obs, chk_a.language, a.max_steps_a, chk_a, frames, "A")
            rec.update(mode="baseline_A", a_success=ok, a_steps=n)
        else:
            rec["b_already_satisfied_at_start"] = chk_b(env)
            # Phase A: 성공과 무관하게 interrupt_step 까지만 수행 (먼저 끝나면 조기 종료)
            obs, a_pre, n_a = self.run_phase(env, obs, chk_a.language, a.interrupt_step, chk_a,
                                             frames, "A")
            rec.update(mode="interrupt_resume", switch_mode=a.switch_mode,
                       a_done_before_interrupt=a_pre, a_steps_before=n_a,
                       gripper_closed_at_interrupt=gripper_closed(obs))

            if a.switch_mode == "defer" and not a_pre and gripper_closed(obs):
                # 안전 지점(그리퍼 열림)까지 A 계속 수행
                obs, a_pre, n_d = self.run_phase(env, obs, chk_a.language, a.defer_max, chk_a, frames, "A",
                                                 stop_fn=lambda o: not gripper_closed(o))
                rec.update(defer_steps=n_d, a_done_before_interrupt=a_pre)

            # Phase B
            obs = self.switch(env, obs, frames, rec, "to_b")
            obs, b_ok, n_b = self.run_phase(env, obs, chk_b.language, a.max_steps_b, chk_b, frames, "B")
            rec.update(b_success=b_ok, b_steps=n_b, a_satisfied_after_b=chk_a(env))

            # Phase A2 (재개)
            if chk_a(env):
                a2_ok, n_a2 = True, 0
            else:
                obs = self.switch(env, obs, frames, rec, "to_a2")
                obs, a2_ok, n_a2 = self.run_phase(env, obs, chk_a.language, a.max_steps_resume, chk_a,
                                                  frames, "A2")
            rec.update(a_resume_success=a2_ok, a_resume_steps=n_a2,
                       b_still_satisfied_at_end=chk_b(env), both_satisfied_at_end=a2_ok and chk_b(env))

        rec["wall_sec"] = round(time.time() - t0, 1)
        if frames:
            vdir = Path(a.out) / "videos"
            vdir.mkdir(parents=True, exist_ok=True)
            write_video(str(vdir / f"{self.tag()}_ep{ep_idx}.mp4"), np.stack(frames), fps=30)
        env.close()
        return rec

    def tag(self):
        a = self.args
        if a.task_b is None or a.interrupt_step < 0:
            return f"A{a.task_a}_baseline"
        suffix = "" if a.switch_mode == "direct" else f"_{a.switch_mode}"
        return f"A{a.task_a}_B{a.task_b}_int{a.interrupt_step}{suffix}"


def summarize(recs):
    keys = ["a_success", "a_done_before_interrupt", "gripper_closed_at_interrupt", "b_success",
            "a_satisfied_after_b", "a_resume_success", "b_still_satisfied_at_end", "both_satisfied_at_end"]
    out = {"n": len(recs)}
    for k in keys:
        vals = [r[k] for r in recs if k in r]
        if vals:
            out[k + "_rate"] = round(float(np.mean(vals)), 3)
    return out


def main():
    args = parse_args()
    runner = Runner(args)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    recs = []
    for ep in range(args.start_episode, args.start_episode + args.episodes):
        r = runner.episode(ep)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        recs.append(r)
    summary = summarize(recs)
    print("SUMMARY", json.dumps(summary, ensure_ascii=False))
    with open(Path(args.out) / f"{runner.tag()}.json", "w") as f:
        json.dump({"args": vars(args), "summary": summary, "episodes": recs}, f, ensure_ascii=False,
                  indent=2)


if __name__ == "__main__":
    main()
