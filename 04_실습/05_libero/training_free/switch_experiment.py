#!/usr/bin/env python
"""
LIBERO-Goal + SmolVLA: 태스크 도중 지시 전환 실험 (연구주제.md / LAB05 7단계)

한 에피소드:
  Phase A  : 태스크 A 지시로 수행 → 전환 시점(--switch-at)에 도달
  Phase B  : 환경 리셋 없이 지시를 B 로 교체 (--strategy 로 남은 chunk 처리) → B 성공 or 시간초과
  Phase A2 : 다시 A 로 교체 (같은 전략) → A 성공 or 시간초과   (재개 실험)

전환 시점 (--switch-at)
  step:N   A 시작 후 N 스텝 (예: step:15 = 접근 중)
  grasp:K  처음 물체를 잡은 뒤 K 스텝 (예: grasp:3 = 잡은 직후, grasp:20 = 들고 이동 중)

전략 (--strategy)  ※ action chunk 단위로 설계 → 로봇/모델 종류와 무관
  none     전환하지 않음 (A 만 계속). 같은 시점의 저크/낙하 기준값
  flush    남은 chunk 버리고 즉시 새 지시로 재추론
  keep     남은 chunk(최대 n_action_steps-1 스텝)를 끝까지 실행한 뒤 새 지시로 추론
  blend    새 지시 chunk 와 이전 chunk 의 겹치는 앞부분 k 스텝을 선형 가중평균
  retreat  전환 안전 처리: 쥔 물체 내려놓기 → 그리퍼 열기 → 들어올리기 → 초기 자세 복귀 → flush
  release  내려놓기 + 그리퍼 열기까지만 (자세 복귀 없음)        ← retreat ablation
  ret_pos  내려놓기 + 위치만 초기값으로 복귀 (손목 회전은 그대로) ← retreat ablation
  ret_rot  내려놓기 + 손목 회전만 초기값으로 복귀 (위치는 그대로) ← retreat ablation
  rtc      Real-Time Chunking (Black et al., NeurIPS 2025). 새 chunk 를 만들 때마다 이전 chunk 의 남은 동작을
           guidance 로 주어 이어지게 생성(inpainting). 전환 순간에도 같은 방식 → 학습 없이 매끄러운 전환
  flush_rtc 전환 순간엔 이전 chunk 를 버리고(flush), 새 지시의 첫 chunk 이후부터 RTC 로 이어붙임.
           rtc 가 전환 때 A 방향으로 끌려가는 문제(2026-09-17)를 피하면서 B 안에서는 매끄럽게
  bon      Best-of-N (Q-Planning, arXiv 2608.21204 의 축소판). 전환 이후 매 추론마다 후보 chunk N개를 뽑고,
           "새 태스크를 성공한 궤적들이 지나간 곳"에 가장 가까이 가는 후보를 고른다. 초기 자세로 돌아가지 않고
           성공 궤적의 길목으로 합류하게 하는 학습 없는 선택 (data/success_manifold.npz 필요)
  vbon     bon 과 같이 후보 N개를 뽑되, 점수를 **학습한 판정기 Q(상황, 동작)** 로 매긴다 (V-GPS / Q-Planning 방식).
           train_value.py 로 만든 모델 필요 (--value-model)

지표 (전환 시점 기준)
  b_success, a_resume_success, jerk, drop, b_failure_type  (+ 반응 시간은 analyze.py)

예시:
  python switch_experiment.py --task-a 8 --task-b 7 --switch-at grasp:3 --strategy blend --episodes 10 --video
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
DT = 1 / 20  # LIBERO control_freq = 20Hz
GRIPPER_OPEN_QPOS = 0.035  # 손가락 qpos 가 이보다 작으면 닫힘
HOLD_DIST = 0.10  # 그리퍼 닫힘 + 끝단에서 이 거리(m) 이내의 물체 → 잡고 있다고 판정
POS_SCALE, ROT_SCALE = 0.05, 0.5  # OSC_POSE delta 액션 1.0 당 이동(m) / 회전(rad)
COLORS = {"A": (60, 120, 255), "B": (255, 150, 30), "A2": (40, 200, 90), "R": (160, 160, 160)}


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def default_args():
    """모든 옵션의 기본값. 다른 스크립트가 일부 옵션만 넘겨도 빠진 건 여기서 채운다."""
    return build_parser().parse_args(["--task-a", "0"])


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="HuggingFaceVLA/smolvla_libero")
    p.add_argument("--task-a", type=int, required=True)
    p.add_argument("--task-b", type=int, default=None)
    p.add_argument("--switch-at", default="grasp:3", help="step:N | grasp:K")
    p.add_argument("--strategy",
                   choices=["none", "flush", "keep", "blend", "retreat", "release", "ret_pos", "ret_rot", "rtc", "flush_rtc", "bon", "vbon"],
                   default="flush")
    p.add_argument("--blend-steps", type=int, default=10)
    p.add_argument("--rtc-horizon", type=int, default=10, help="rtc: 이전 chunk 를 따르도록 유도할 앞부분 길이")
    p.add_argument("--rtc-delay", type=int, default=0, help="rtc: 완전히 고정할 앞부분 길이 (추론 지연 모사)")
    p.add_argument("--rtc-guidance", type=float, default=10.0, help="rtc: 최대 guidance 가중치")
    p.add_argument("--bon-n", type=int, default=8, help="bon: 후보 chunk 개수")
    p.add_argument("--bon-horizon", type=int, default=20, help="bon: 점수 매길 때 내다볼 스텝 수")
    p.add_argument("--bon-gain", type=float, default=0.6, help="bon: delta 동작 → 실제 이동 근사 배율")
    p.add_argument("--bon-batch", type=int, default=2, help="bon: 후보를 한 번에 몇 개씩 계산할지 (1080 Ti 메모리 한계)")
    p.add_argument("--manifold", default="data/success_manifold.npz")
    p.add_argument("--value-model", default="outputs/value_model/q_model.pt", help="vbon: 판정기 경로")
    p.add_argument("--log-value", action="store_true",
                   help="전환 이후 매 추론마다 (이미지·지시 특징, 로봇 상태, 실행한 동작)을 저장 — 판정기(가치 함수) 학습용")
    p.add_argument("--value-horizon", type=int, default=10, help="--log-value: 저장할 동작 앞부분 길이")
    p.add_argument("--n-action-steps", type=int, default=10, help="chunk 에서 몇 스텝 실행 후 재추론 (통제변수)")
    p.add_argument("--max-steps", type=int, default=300, help="각 단계(A, B, A2) 최대 스텝")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--start-episode", type=int, default=0, help="LIBERO 고정 초기상태 인덱스")
    p.add_argument("--video", action="store_true")
    p.add_argument("--out", default="outputs/switch")
    p.add_argument("--seed", type=int, default=7)
    return p


def add_batch_dim(tree):
    if isinstance(tree, dict):
        return {k: add_batch_dim(v) for k, v in tree.items()}
    return np.asarray(tree)[None]


def gripper_closed(obs) -> bool:
    return bool(obs["robot_state"]["gripper"]["qpos"][0] < GRIPPER_OPEN_QPOS)


def eef_pos(obs) -> np.ndarray:
    return np.asarray(obs["robot_state"]["eef"]["pos"], dtype=np.float64)


def jerk_score(traj, dt=DT):
    """LAB03 과 같은 정의: 3차 차분 크기의 평균. 작을수록 부드러움."""
    traj = np.asarray(traj)
    if len(traj) < 4:
        return None
    jrk = np.diff(traj, n=3, axis=0) / dt**3
    return float(np.linalg.norm(jrk, axis=1).mean())


class GoalChecker:
    """임의의 LIBERO_GOAL 태스크 goal 을 현재 시뮬레이션 상태에서 평가 (같은 장면을 공유하므로 가능)."""

    def __init__(self, suite, task_id: int):
        task = suite.get_task(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        self.goal = robosuite_parse_problem(bddl)["goal_state"]
        self.language = task.language

    def __call__(self, env: LiberoEnv) -> bool:
        inner = env._env.env
        return all(inner._eval_predicate(s) for s in self.goal)


class Episode:
    """한 에피소드의 환경 + 정책 실행 상태 + 기록."""

    def __init__(self, runner, ep_idx):
        a = runner.args
        self.r, self.a = runner, a
        self.env = LiberoEnv(task_suite=runner.suite, task_id=a.task_a, task_suite_name=SUITE,
                             obs_type="pixels_agent_pos", episode_index=ep_idx)
        torch.manual_seed(a.seed + ep_idx)
        self.obs, _ = self.env.reset(seed=a.seed + ep_idx)
        self.inner = self.env._env.env
        self.home = (eef_pos(self.obs).copy(), self.obs["robot_state"]["eef"]["mat"].copy())
        self.plan = np.zeros((0, 7))  # 현재 chunk 에서 아직 실행 안 한 동작들
        self.plan_norm = None  # 같은 동작의 정규화 값 (RTC guidance 에 필요)
        self.bon_active = False  # bon: 전환 이후에만 후보 선택
        self.rtc_active = False  # rtc: 전환 이후에만 guidance
        self.exec_left = 0  # 재추론 전까지 더 실행할 스텝 수
        self.frames = [] if a.video else None
        self.log = {"pos": [], "gq": [], "phase": [], "held_z": []}
        self.bon_log = []
        self.value_log = []  # --log-value: 전환 이후 결정마다 기록
        self.value_active = False
        self.held = None  # 잡고 있는 물체 이름

    # ---------- 정책 ----------
    def infer_chunk(self, instruction) -> np.ndarray:
        r = self.r
        use_rtc = self.a.strategy in ("rtc", "flush_rtc") and self.rtc_active  # 전환 이후에만
        # RTC 는 guidance 계산에 autograd 를 쓰므로 inference_mode 대신 no_grad
        ctx = torch.no_grad() if use_rtc else torch.inference_mode()
        with ctx:
            batch = preprocess_observation(add_batch_dim(self.obs))
            batch["task"] = [instruction]
            batch = r.pre(r.env_step(batch))
            kwargs = {}
            if use_rtc and self.plan_norm is not None and len(self.plan_norm) > 0:
                kwargs = {"prev_chunk_left_over": self.plan_norm.unsqueeze(0),
                          "inference_delay": self.a.rtc_delay, "execution_horizon": self.a.rtc_horizon}
            if self.bon_active:
                return self.best_of_n(batch, instruction)
            norm = r.policy.predict_action_chunk(batch, **kwargs)  # (1, chunk_size, 7) 정규화 공간
            chunk = r.post(norm.clone())
        self.plan_norm = norm[0].detach()
        out = chunk[0].float().cpu().numpy()
        self.record_value(instruction, out)
        return out

    def record_value(self, instruction, chunk, candidates=None):
        """판정기 학습용: 방금 계산한 prefix 특징(이미지+지시+상태), 로봇 상태, 실행할 동작 앞부분."""
        if not (self.a.log_value and self.value_active and self.r.last_prefix_feat is not None):
            return
        h = self.a.value_horizon
        o = self.obs["robot_state"]
        self.value_log.append({
            "step": len(self.log["pos"]),
            "feat": self.r.last_prefix_feat[0].float().cpu().numpy().astype(np.float16),
            "state": np.concatenate([o["eef"]["pos"], o["eef"]["quat"], o["gripper"]["qpos"]]).astype(np.float32),
            "chunk": chunk[:h].astype(np.float32),
            "candidates": None if candidates is None else candidates[:, :h].astype(np.float32),
            "instruction": instruction,
        })

    def best_of_n(self, batch, instruction):
        """후보 N개 중 '성공 궤적 근처로 가는' 후보 선택. 점수 = 예상 끝단 위치와 성공 궤적 점들 사이 거리 평균."""
        r, n = self.r, self.a.bon_n
        norms = []
        for i in range(0, n, self.a.bon_batch):  # 한 번에 다 하면 이미지 인코더가 N배 메모리를 씀
            m = min(self.a.bon_batch, n - i)
            rep = {k: (v.repeat(m, *([1] * (v.dim() - 1))) if torch.is_tensor(v) and v.shape[:1] == (1,) else
                       (v * m if isinstance(v, list) and len(v) == 1 else v)) for k, v in batch.items()}
            norms.append(r.policy.predict_action_chunk(rep))  # 노이즈가 달라 후보가 서로 다름
        norm = torch.cat(norms, 0)  # (N, T, 7)
        chunks = r.post(norm.clone()).float().cpu().numpy()
        tree = r.manifold.get(instruction)
        if self.a.strategy == "vbon":
            h = self.a.value_horizon
            o = self.obs["robot_state"]
            state = np.concatenate([o["eef"]["pos"], o["eef"]["quat"], o["gripper"]["qpos"]]).astype(np.float32)
            feat = r.last_prefix_feat[0].float().cpu().numpy()
            X = np.stack([np.concatenate([feat, state, c[:h].reshape(-1)]) for c in chunks]).astype(np.float32)
            with torch.no_grad():
                q = r.qnet(torch.tensor(X, device=r.device)).cpu().numpy()
            best = int(np.argmax(q))
            self.bon_log.append({"step": len(self.log["pos"]), "best": round(float(q[best]), 3),
                                 "worst": round(float(q.min()), 3), "median": round(float(np.median(q)), 3)})
        elif tree is None:
            best = 0
        else:
            h = min(self.a.bon_horizon, chunks.shape[1])
            p0 = eef_pos(self.obs)
            scores = []
            for c in chunks:
                traj = p0 + np.cumsum(c[:h, :3], axis=0) * POS_SCALE * self.a.bon_gain
                d, _ = tree.query(traj)
                scores.append(float(d.mean()))
            best = int(np.argmin(scores))
            self.bon_log.append({"step": len(self.log["pos"]), "best": round(scores[best] * 100, 2),
                                 "worst": round(max(scores) * 100, 2), "median": round(float(np.median(scores)) * 100, 2)})
        self.plan_norm = norm[best].detach()
        self.record_value(instruction, chunks[best], candidates=chunks)
        return chunks[best]

    def policy_action(self, instruction):
        if self.exec_left <= 0 or len(self.plan) == 0:
            self.plan = self.infer_chunk(instruction)
            self.exec_left = self.a.n_action_steps
        act, self.plan = self.plan[0], self.plan[1:]
        if self.plan_norm is not None:
            self.plan_norm = self.plan_norm[1:]
        self.exec_left -= 1
        return act

    RETREAT_MODES = {"retreat": (True, True), "release": (False, False),
                     "ret_pos": (True, False), "ret_rot": (False, True)}

    def apply_strategy(self, new_instruction, rec, key):
        s = self.a.strategy
        self.value_active = True
        if s in ("flush",) or s in self.RETREAT_MODES:
            if s in self.RETREAT_MODES:
                restore_pos, restore_rot = self.RETREAT_MODES[s]
                rec[f"{key}_retreat_steps"] = self.retreat(restore_pos, restore_rot)
            self.plan, self.exec_left = np.zeros((0, 7)), 0
        elif s in ("bon", "vbon"):
            self.plan, self.exec_left = np.zeros((0, 7)), 0
            self.bon_active = True
        elif s == "flush_rtc":
            # 전환 순간: 이전(A) chunk 를 완전히 버림 → 첫 B chunk 는 guidance 없이 자유롭게, 그 뒤부터 RTC
            self.plan, self.exec_left, self.plan_norm = np.zeros((0, 7)), 0, None
            self.rtc_active = True
        elif s == "rtc":
            # 남은 동작(plan_norm)은 그대로 두고 즉시 새 지시로 재추론 → infer_chunk 가 guidance 로 이어붙임
            self.rtc_active = True
            self.exec_left = 0
        elif s == "keep":
            pass  # 남은 exec_left 스텝을 실행한 뒤 자연스럽게 새 지시로 재추론
        elif s == "blend":
            new = self.infer_chunk(new_instruction)
            old = self.plan
            n = min(self.a.blend_steps, len(old))
            w = ((np.arange(n) + 1) / (n + 1))[:, None]
            self.plan = np.concatenate([(1 - w) * old[:n] + w * new[:n], new[n:]])
            self.exec_left = self.a.n_action_steps

    # ---------- 환경 ----------
    def step(self, action, phase):
        raw, _, _, _ = self.env._env.step(np.asarray(action, dtype=np.float64))
        self.obs = self.env._format_raw_obs(raw)
        self.log["pos"].append(eef_pos(self.obs))
        self.log["gq"].append(float(self.obs["robot_state"]["gripper"]["qpos"][0]))
        self.log["phase"].append(phase)
        self.log["held_z"].append(self.obj_pos(self.held)[2] if self.held else np.nan)
        if self.frames is not None:
            img = self.obs["pixels"]["image"][::-1, ::-1]
            wrist = self.obs["pixels"]["image2"][::-1, ::-1]
            f = np.concatenate([img, wrist], axis=1).copy()
            f[:8] = COLORS[phase]
            self.frames.append(f)

    def obj_pos(self, name):
        return np.array(self.inner.sim.data.body_xpos[self.inner.obj_body_id[name]])

    def nearest_object(self):
        p = eef_pos(self.obs)
        dists = {n: np.linalg.norm(self.obj_pos(n) - p) for n in self.inner.objects_dict}
        name = min(dists, key=dists.get)
        return name if dists[name] < HOLD_DIST else None

    def holding(self):
        return gripper_closed(self.obs) and self.nearest_object() is not None

    def run_policy(self, instruction, phase, max_steps, success_fn=None, trigger_fn=None, watch_fn=None):
        """success_fn/trigger_fn 이 True 가 되거나 max_steps 까지. (끝난 이유, 스텝 수) 반환."""
        for t in range(max_steps):
            self.step(self.policy_action(instruction), phase)
            if watch_fn is not None:
                watch_fn()
            if success_fn is not None and success_fn(self.env):
                return "success", t + 1
            if trigger_fn is not None and trigger_fn(t + 1):
                return "trigger", t + 1
        return "timeout", max_steps

    def retreat(self, restore_pos=True, restore_rot=True):
        """쥔 물체 내려놓기 → 그리퍼 열기 → (들어올리기 → 초기 위치/회전 복귀). 사용한 스텝 수 반환.

        restore_pos / restore_rot 로 무엇을 복귀시킬지 나눌 수 있다 (ablation).
        """
        home_pos, home_mat = self.home
        n0 = len(self.log["pos"])
        if gripper_closed(self.obs):
            zs = []
            for _ in range(40):  # 접촉으로 하강이 멈출 때까지 (공중에서 떨어뜨리지 않도록)
                self.step([0, 0, -0.3, 0, 0, 0, 1], "R")
                zs.append(eef_pos(self.obs)[2])
                if len(zs) >= 5 and zs[-5] - zs[-1] < 0.003:
                    break
        for _ in range(15):
            self.step([0, 0, 0, 0, 0, 0, -1], "R")
        if restore_pos:  # 물체와 부딪히지 않게 먼저 들어올림
            for _ in range(30):
                if eef_pos(self.obs)[2] >= home_pos[2] - 0.01:
                    break
                self.step([0, 0, 0.6, 0, 0, 0, -1], "R")
        if restore_pos or restore_rot:
            for _ in range(100):
                dp = (home_pos - eef_pos(self.obs)) if restore_pos else np.zeros(3)
                rot_err = (Rotation.from_matrix(home_mat @ self.obs["robot_state"]["eef"]["mat"].T).as_rotvec()
                           if restore_rot else np.zeros(3))
                if np.linalg.norm(dp) < 0.01 and np.linalg.norm(rot_err) < 0.05:
                    break
                self.step(np.concatenate([np.clip(dp / POS_SCALE, -1, 1), np.clip(rot_err / ROT_SCALE, -1, 1), [-1]]),
                          "R")
        for _ in range(5):
            self.step(get_libero_dummy_action(), "R")
        return len(self.log["pos"]) - n0


def switch_metrics(log, ts, b_end, held):
    """전환 시점 ts(로그 인덱스) 기준 저크·낙하.
    반응 시간은 같은 에피소드의 strategy=none 궤적과 비교해야 하므로 analyze.py 에서 계산한다
    (전환 전까지 궤적이 완전히 동일함을 확인함)."""
    pos = np.asarray(log["pos"])
    gq = np.asarray(log["gq"])
    out = {}
    out["jerk"] = jerk_score(pos[max(0, ts - 5): ts + 45])
    # 낙하: 전환 후 B 구간 안에서 그리퍼가 열린 순간 물체 높이 - 15스텝 뒤 높이 > 3cm
    out["drop"] = False
    if held is not None:
        z = np.asarray(log["held_z"])
        for t in range(max(ts, 1), b_end):
            if gq[t - 1] < GRIPPER_OPEN_QPOS <= gq[t]:
                after = z[min(t + 15, len(z) - 1)]
                if z[t] - after > 0.03:
                    out["drop"] = True
                    break
    return out


def fill_defaults(args):
    """다른 스크립트가 만든 옵션 묶음에 빠진 항목을 기본값으로 채운다 (새 옵션이 추가돼도 안 깨지게)."""
    for k, v in vars(default_args()).items():
        if not hasattr(args, k):
            setattr(args, k, v)
    return args


class Runner:
    def __init__(self, args):
        self.args = fill_defaults(args)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = SmolVLAPolicy.from_pretrained(args.policy)
        self.policy.config.device = self.device
        if getattr(args, "strategy", None) in ("rtc", "flush_rtc"):
            from lerobot.policies.rtc.configuration_rtc import RTCConfig
            self.policy.config.rtc_config = RTCConfig(enabled=True, execution_horizon=args.rtc_horizon,
                                                      max_guidance_weight=args.rtc_guidance)
            self.policy.init_rtc_processor()
        self.policy.to(self.device).eval()
        self.last_prefix_feat = None
        self.qnet = None
        if getattr(args, "strategy", None) == "vbon":
            from train_value import MLP
            ck = torch.load(args.value_model, map_location=self.device)
            self.qnet = MLP(ck["d_in"], ck["hidden"]).to(self.device).eval()
            self.qnet.load_state_dict(ck["state_dict"])
        if getattr(args, "log_value", False) or getattr(args, "strategy", None) == "vbon":
            orig = self.policy.model.embed_prefix

            def embed_prefix_and_keep(*a_, **kw):
                embs, pad, att = orig(*a_, **kw)
                m = pad.to(embs.dtype).unsqueeze(-1)
                self.last_prefix_feat = ((embs * m).sum(1) / m.sum(1).clamp(min=1)).detach()
                return embs, pad, att

            self.policy.model.embed_prefix = embed_prefix_and_keep
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.policy.config, pretrained_path=args.policy,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )
        self.env_step = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
        self.suite = benchmark.get_benchmark_dict()[SUITE]()
        self.manifold = {}
        if getattr(args, "strategy", None) == "bon":
            from scipy.spatial import cKDTree
            z = np.load(args.manifold, allow_pickle=True)
            for i, task in enumerate(z["tasks"]):
                self.manifold[str(task)] = cKDTree(z[f"t{i}"])
        self.chk_a = GoalChecker(self.suite, args.task_a)
        self.chk_b = GoalChecker(self.suite, args.task_b) if args.task_b is not None else None
        kind, val = args.switch_at.split(":")
        self.switch_kind, self.switch_val = kind, int(val)

    def tag(self):
        a = self.args
        b = f"B{a.task_b}" if a.task_b is not None else "Bnone"
        return f"A{a.task_a}_{b}_{a.switch_at.replace(':', '')}_{a.strategy}"

    def episode(self, ep_idx):
        a = self.args
        ep = Episode(self, ep_idx)
        self.policy.reset()
        rec = {"episode": ep_idx, "task_a": self.chk_a.language,
               "task_b": self.chk_b.language if self.chk_b else None,
               "switch_at": a.switch_at, "strategy": a.strategy}
        t0 = time.time()

        # ---- Phase A: 전환 시점까지 ----
        grasp_step = {"t": None}

        def watch():
            if grasp_step["t"] is None and ep.holding():
                grasp_step["t"] = len(ep.log["pos"])

        def trigger(t):
            if self.switch_kind == "step":
                return t >= self.switch_val
            return grasp_step["t"] is not None and t >= grasp_step["t"] + self.switch_val

        why, n_a = ep.run_policy(self.chk_a.language, "A", a.max_steps, self.chk_a, trigger, watch)
        rec.update(grasp_step=grasp_step["t"], a_steps_before_switch=n_a)
        if why != "trigger":
            # A 가 전환 전에 끝났거나(성공) 잡지 못한 채 시간초과 → 전환 조건 미충족, 집계 제외
            rec.update(switched=False, a_success=why == "success")
            return self.finish(ep, rec, t0, ep_idx)

        ts = len(ep.log["pos"])
        ep.held = ep.nearest_object() if gripper_closed(ep.obs) else None
        rec.update(switched=True, holding_at_switch=ep.held is not None, held_object=ep.held,
                   gripper_closed_at_switch=gripper_closed(ep.obs))
        # 낙하 판정용 높이 로그를 전환 시점부터 채움
        ep.log["held_z"][-1] = ep.obj_pos(ep.held)[2] if ep.held else np.nan

        if a.strategy == "none" or self.chk_b is None:
            why, n = ep.run_policy(self.chk_a.language, "A", a.max_steps, self.chk_a)
            rec.update(a_success=why == "success", a_total_steps=n_a + n)
            rec.update(switch_metrics(ep.log, ts, len(ep.log["pos"]), ep.held))
            return self.finish(ep, rec, t0, ep_idx)

        # ---- Phase B ----
        a_done_at_switch = self.chk_a(ep.env)
        rec["b_satisfied_at_switch"] = self.chk_b(ep.env)
        a_during_b = {"v": False}

        def watch_b():
            if not a_done_at_switch and self.chk_a(ep.env):
                a_during_b["v"] = True

        ep.apply_strategy(self.chk_b.language, rec, "to_b")
        why, n_b = ep.run_policy(self.chk_b.language, "B", a.max_steps, self.chk_b, watch_fn=watch_b)
        b_end = len(ep.log["pos"])
        rec.update(b_success=why == "success", b_steps=n_b, a_completed_during_b=a_during_b["v"])
        rec.update(switch_metrics(ep.log, ts, b_end, ep.held))

        pos = np.asarray(ep.log["pos"])
        b_path = np.linalg.norm(np.diff(pos[ts:min(ts + 60, b_end)], axis=0), axis=1).sum() if b_end - ts > 1 else 0
        if rec["b_success"]:
            rec["b_failure_type"] = None
        elif rec["drop"]:
            rec["b_failure_type"] = "물체 낙하"
        elif a_during_b["v"]:
            rec["b_failure_type"] = "지시 무시"
        elif b_path < 0.03:
            rec["b_failure_type"] = "얼어붙음"
        else:
            rec["b_failure_type"] = "시간 초과"

        # ---- Phase A2: 재개 ----
        if self.chk_a(ep.env):
            ok, n_a2 = True, 0
        else:
            ep.apply_strategy(self.chk_a.language, rec, "to_a2")
            why, n_a2 = ep.run_policy(self.chk_a.language, "A2", a.max_steps, self.chk_a)
            ok = why == "success"
        rec.update(a_resume_success=ok, a_resume_steps=n_a2, b_still_satisfied_at_end=self.chk_b(ep.env))
        rec["both_success"] = rec["b_success"] and ok and rec["b_still_satisfied_at_end"]
        return self.finish(ep, rec, t0, ep_idx)

    def finish(self, ep, rec, t0, ep_idx):
        rec["wall_sec"] = round(time.time() - t0, 1)
        if ep.value_log:
            vd = Path(self.args.out) / "value"
            vd.mkdir(parents=True, exist_ok=True)
            phase = np.asarray(ep.log["phase"])
            b_idx = np.nonzero(phase == "B")[0]
            a2_idx = np.nonzero(phase == "A2")[0]
            np.savez_compressed(
                vd / f"{self.tag()}_ep{ep_idx}.npz",
                step=np.array([v["step"] for v in ep.value_log]),
                feat=np.stack([v["feat"] for v in ep.value_log]),
                state=np.stack([v["state"] for v in ep.value_log]),
                chunk=np.stack([v["chunk"] for v in ep.value_log]),
                instruction=np.array([v["instruction"] for v in ep.value_log], dtype=object),
                has_candidates=np.array([v["candidates"] is not None for v in ep.value_log]),
                b_success=bool(rec.get("b_success")), a_resume_success=bool(rec.get("a_resume_success")),
                b_end=int(b_idx[-1] + 1) if len(b_idx) else -1, a2_end=int(a2_idx[-1] + 1) if len(a2_idx) else -1,
                task_b=str(rec.get("task_b")), task_a=str(rec.get("task_a")))
        if ep.bon_log:
            rec["bon_choices"] = len(ep.bon_log)
            rec["bon_best_cm_median"] = float(np.median([b["best"] for b in ep.bon_log]))
            rec["bon_median_cm_median"] = float(np.median([b["median"] for b in ep.bon_log]))
        out = Path(self.args.out)
        (out / "traj").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "traj" / f"{self.tag()}_ep{ep_idx}.npz", pos=np.asarray(ep.log["pos"]),
                            gripper_qpos=np.asarray(ep.log["gq"]), phase=np.asarray(ep.log["phase"]),
                            held_z=np.asarray(ep.log["held_z"]))
        if ep.frames:
            (out / "videos").mkdir(parents=True, exist_ok=True)
            write_video(str(out / "videos" / f"{self.tag()}_ep{ep_idx}.mp4"), np.stack(ep.frames), fps=30)
        ep.env.close()
        return rec


def summarize(recs):
    sw = [r for r in recs if r.get("switched")]
    out = {"n": len(recs), "n_switched": len(sw)}
    for k in ["holding_at_switch", "a_success", "b_success", "a_resume_success", "both_success", "drop"]:
        vals = [r[k] for r in sw if r.get(k) is not None]
        if vals:
            out[k + "_rate"] = round(float(np.mean(vals)), 3)
    for k in ["jerk"]:
        vals = [r[k] for r in sw if r.get(k) is not None]
        if vals:
            out[k + "_median"] = round(float(np.median(vals)), 2)
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
    print("SUMMARY", runner.tag(), json.dumps(summary, ensure_ascii=False), flush=True)
    meta = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    with open(Path(args.out) / f"{runner.tag()}.json", "w") as f:
        json.dump({"args": vars(args), "meta": meta, "summary": summary, "episodes": recs}, f,
                  ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
