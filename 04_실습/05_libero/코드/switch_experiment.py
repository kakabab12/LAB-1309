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
  rollback 최근 동작을 거꾸로 재생해 집었던 자리에 내려놓기 (SwitchVLA rollback 방식).
           초기 자세로 가지 않는다 — 되돌리는 범위가 '집기 직전'까지로 제한됨
  rtc      Real-Time Chunking (Black et al., NeurIPS 2025). 새 chunk 를 만들 때마다 이전 chunk 의 남은 동작을
           guidance 로 주어 이어지게 생성(inpainting). 전환 순간에도 같은 방식 → 학습 없이 매끄러운 전환
  rtc_all  처음부터 모든 chunk 경계에 RTC (전환 없는 평범한 수행에서 RTC 자체 영향 확인용, --switch-at step:9999 와 함께)
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
import pathlib
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

import instr_cfg

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
                   choices=["none", "flush", "keep", "blend", "retreat", "release", "ret_pos", "ret_rot",
                            "rollback", "ret_part", "finish_a", "rtc", "flush_rtc", "rtc_all",
                            "bon", "vbon"],
                   default="flush")
    p.add_argument("--blend-steps", type=int, default=10)
    p.add_argument("--finish-a-max", type=int, default=150,
                   help="finish_a: 새 지시를 받고도 A 를 마저 할 때 최대 몇 스텝까지 기다릴지")
    p.add_argument("--retreat-frac", type=float, default=0.5,
                   help="ret_part: 초기 자세 쪽으로 되돌리는 비율 (0=제자리, 1=완전 복귀=retreat). "
                        "얼마나 가까이 가야 하는지를 재기 위한 용량-반응 실험용")
    p.add_argument("--rollback-pre", type=int, default=10,
                   help="rollback: 집기 시작보다 몇 스텝 더 앞까지 되돌릴지")
    p.add_argument("--rollback-max", type=int, default=60,
                   help="rollback: 되돌릴 최대 스텝 수 (초기 자세까지 가지 않도록 제한)")
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
    p.add_argument("--dither-escape", default="off",
                   choices=["off", "home", "lift", "reverse", "random"],
                   help="제자리 맴돌기(idling)를 감지하면 흔들어서 탈출. PIP(arXiv 2508.15669) 참고. "
                        "home 은 초기 자세를 참조하므로 **제약 위반(비교 기준용)**")
    p.add_argument("--dither-window", type=int, default=40, help="idling 감지 창 (스텝, 20=1초)")
    p.add_argument("--dither-range-cm", type=float, default=3.0,
                   help="창 안의 끝단 이동 범위가 이보다 작으면 idling")
    p.add_argument("--dither-steps", type=int, default=10, help="한 번 흔들 때 몇 스텝 동안")
    p.add_argument("--dither-gain", type=float, default=0.5, help="흔드는 세기 (동작 단위, 1.0=5cm/스텝)")
    p.add_argument("--dither-cooldown", type=int, default=40, help="흔든 뒤 다시 흔들기까지 최소 스텝")
    p.add_argument("--switch-noise-seed", type=int, default=-1,
                   help="전환 순간에 flow matching 노이즈 시드를 이 값으로 고정 (-1 이면 고정 안 함). "
                        "오라클에서 '어떤 노이즈는 체계적으로 더 좋다'가 확인되면 그 시드를 쓴다")
    p.add_argument("--noise-shift", default=None,
                   help="전환 이후 flow matching 노이즈의 **평균을 옮긴다** (.npy 경로, 길이 32 또는 50x32). "
                        "노이즈를 고정하는 게 아니라 분포 중심만 옮기므로 변화는 유지된다 "
                        "— 매 추론마다 같은 노이즈를 쓰면 0% 가 되기 때문 (2026-09-18 측정)")
    p.add_argument("--noise-shift-scale", type=float, default=1.0, help="노이즈 평균 이동의 세기 배율")
    p.add_argument("--grip-latch", type=int, default=0,
                   help="전환 이후 N 스텝 동안 그리퍼를 닫힌 채로 유지한다. "
                        "'지시가 바뀌면 일단 놓는' 반사를 막는다. 0 이면 끔")
    p.add_argument("--b-text", default=None,
                   help="B 구간에서 실제로 넣을 지시문을 직접 지정한다 (성공 판정은 --task-b 그대로). "
                        "뜻 없는 글자를 넣어 '정책이 지시 내용을 듣는가, 바뀐 것만 아는가'를 가른다")
    p.add_argument("--instr-repeat", type=int, default=1,
                   help="지시문을 몇 번 반복해 넣을지. 언어 토큰 수를 늘려 주의를 끄는 "
                        "가장 단순한 증폭. 1 이면 원래대로")
    p.add_argument("--instr-repeat-from", choices=["switch", "always"], default="switch")
    p.add_argument("--cfg-w", type=float, default=1.0,
                   help="지시문 증폭 계수. v = v_A + w(v_B - v_A). 1.0 이면 원래 정책과 동일")
    p.add_argument("--cfg-from", choices=["switch", "always"], default="switch",
                   help="switch=전환 이후에만 증폭, always=처음부터")
    p.add_argument("--noise-scale", type=float, default=1.0,
                   help="전환 이후 노이즈의 **크기 배율** (1.0 = 원래). 0 에 가까우면 결정적이 되어 0% 가 되고, "
                        "1보다 크면 더 다양한 동작이 나온다 — '변화가 탈출을 돕는가'를 재는 실험")
    p.add_argument("--noise-seed-from-start", action="store_true",
                   help="에피소드 **시작부터** 노이즈 시드를 고정 (전환 없이도 적용). "
                        "'좋은 노이즈가 전환과 무관하게 그냥 그 태스크에 좋은가'를 가른다")
    p.add_argument("--switch-noise-first-only", action="store_true",
                   help="전환 직후 **첫 chunk 에만** 고정 시드를 쓰고 그 뒤는 무작위로 되돌린다. "
                        "'첫 동작 묶음이 결과를 결정하는가'를 가르는 ablation")
    p.add_argument("--switch-noise-reseed", action="store_true",
                   help="전환 이후 **매 추론마다** 같은 시드로 다시 seed → 정책이 완전히 결정적이 된다. "
                        "'특정 노이즈 순서가 좋은가' vs '결정적인 것 자체가 좋은가'를 가른다")
    p.add_argument("--switch-n-action-steps", type=int, default=0,
                   help="전환 이후에만 쓰는 재추론 간격 (0 이면 --n-action-steps 와 같음). "
                        "작게 하면 나쁜 후보에 갇히는 시간이 줄어든다 — 오라클 결과의 후속 실험")
    p.add_argument("--latency-steps", type=int, default=0,
                   help="추론 지연 모사 (스텝, 20스텝=1초). 요청한 chunk 가 이만큼 뒤에 도착하고, 그동안 이전 계획을 계속 실행. "
                        "도착한 chunk 는 지나간 앞부분을 버리고 이어서 씀. 0 이면 기존 동기 실행")
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
        self.pending = None  # 지연 모사: [동작, 정규화 동작, 남은 도착 스텝]
        self.last_grip = -1.0
        self.hold_steps = 0
        self.rtc_active = False  # rtc: 전환 이후에만 guidance
        self.exec_left = 0  # 재추론 전까지 더 실행할 스텝 수
        self.n_act = a.n_action_steps  # 재추론 간격 (전환 이후 바뀔 수 있음)
        self.frames = [] if a.video else None
        self.log = {"pos": [], "gq": [], "phase": [], "held_z": [], "rot": []}
        self.acts = []  # 실제로 실행한 동작 (rollback: 거꾸로 되돌리기 위해)
        self.bon_log = []
        self.value_log = []  # --log-value: 전환 이후 결정마다 기록
        self.value_active = False
        self.escape_rng = np.random.default_rng(a.seed + 977 * ep_idx)
        self.record = None  # 학습 데이터 수집용 훅: record(obs_before, action, phase)
        self.ep_idx = ep_idx
        self.post_switch_infers = 0  # 전환 이후 추론 횟수 (first-only ablation 용)
        if a.noise_seed_from_start and a.switch_noise_seed >= 0:
            torch.manual_seed(a.switch_noise_seed)  # 전환과 무관하게 처음부터 고정
        self.held = None  # 잡고 있는 물체 이름
        self.grasp_t = None  # 그리퍼가 마지막으로 닫힌 시점 (rollback 되돌림 범위 제한용)
        self.was_closed = False  # 그리퍼 열림→닫힘 전이 감지용
        self.escape_left = 0  # 남은 흔들기 스텝
        self.latch_left = 0  # 남은 그리퍼 유지 스텝
        self.escape_act = None  # 흔들 때 내보낼 동작
        self.escape_cool = 0  # 흔들기 쿨다운
        self.escape_count = 0  # 몇 번 흔들었나
        self.escape_at = []  # 흔든 시점들

    # ---------- 정책 ----------
    def maybe_repeat(self, instruction):
        """지시문 반복 — 언어 토큰 수를 늘려 주의를 끈다.

        CMI 가 쥔 직후 0.284 로 떨어지는 것이 '언어가 밀려나서' 라면,
        같은 말을 여러 번 넣어 언어 토큰의 비중을 키우는 것만으로도 달라질 수 있다.
        속도장을 건드리는 증폭과 달리 **동작 크기가 커지지 않아** 자연스러움에 안전하다.
        """
        k = self.a.instr_repeat
        if k <= 1:
            return instruction
        if not (self.value_active or self.a.instr_repeat_from == "always"):
            return instruction
        s = instruction.rstrip(". ")
        return ". ".join([s] * k) + "."

    def cfg_active(self):
        """지시문 증폭을 지금 켜야 하는가.

        기본은 **전환 이후에만** 켠다 (CMI 가 떨어지는 구간이 거기다).
        w=1.0 이면 원래 정책과 수학적으로 같으므로 아예 건너뛴다.
        """
        if self.a.cfg_w == 1.0 or self.r.chk_a is None:
            return False
        return self.value_active or self.a.cfg_from == "always"

    def infer_chunk(self, instruction, commit=True):
        """commit=False 면 (동작, 정규화 동작)을 돌려주고 현재 계획은 건드리지 않는다 (지연 모사용)."""
        r = self.r
        use_rtc = (self.a.strategy in ("rtc", "flush_rtc") and self.rtc_active) or self.a.strategy == "rtc_all"
        # RTC 는 guidance 계산에 autograd 를 쓰므로 inference_mode 대신 no_grad
        if (self.a.switch_noise_reseed and self.a.switch_noise_seed >= 0
                and (self.value_active or self.a.noise_seed_from_start)):
            torch.manual_seed(self.a.switch_noise_seed)  # 매 추론마다 같은 노이즈
        if self.a.switch_noise_first_only and self.value_active:
            self.post_switch_infers += 1
            if self.post_switch_infers == 2:
                # 첫 chunk 를 뽑은 뒤부터는 무작위로 되돌린다
                torch.manual_seed(int(self.a.seed) * 7919 + 13 * self.ep_idx + 101)
        ctx = torch.no_grad() if use_rtc else torch.inference_mode()
        with ctx:
            batch = preprocess_observation(add_batch_dim(self.obs))
            batch["task"] = [self.maybe_repeat(instruction)]
            batch = r.pre(r.env_step(batch))
            kwargs = {}
            if use_rtc and self.plan_norm is not None and len(self.plan_norm) > 0:
                delay = self.a.latency_steps if self.a.latency_steps > 0 else self.a.rtc_delay
                kwargs = {"prev_chunk_left_over": self.plan_norm.unsqueeze(0),
                          "inference_delay": delay, "execution_horizon": max(self.a.rtc_horizon, delay)}
            if self.bon_active:
                return self.best_of_n(batch, instruction)
            need_noise = (self.a.noise_scale != 1.0 or self.r.noise_shift is not None)
            if need_noise and (self.value_active or self.a.noise_seed_from_start):
                # 노이즈 평균만 옮긴다 (z = μ + ε). 변화가 남아 있어야 진행이 된다
                cfg = r.policy.config
                z = torch.randn(1, cfg.chunk_size, cfg.max_action_dim,
                                device=r.device, dtype=torch.float32) * self.a.noise_scale
                if self.r.noise_shift is not None:
                    z = z + self.r.noise_shift
                kwargs["noise"] = z
            if self.cfg_active():
                # 지시문 증폭: 이전 지시를 음의 조건으로 써서 새 지시를 키운다
                nb = preprocess_observation(add_batch_dim(self.obs))
                # 음의 조건도 같은 횟수로 반복해야 한다 — 한쪽만 길면
                # 토큰 수 차이가 증폭 효과에 섞인다
                nb["task"] = [self.maybe_repeat(self.r.chk_a.language)]
                nb = r.pre(r.env_step(nb))
                norm = instr_cfg.predict_chunk_cfg(r.policy, nb, batch, self.a.cfg_w,
                                                   noise=kwargs.get("noise"))
            else:
                norm = r.policy.predict_action_chunk(batch, **kwargs)  # (1, chunk_size, 7) 정규화 공간
            chunk = r.post(norm.clone())
        out = chunk[0].float().cpu().numpy()
        if not commit:
            return out, norm[0].detach()
        self.plan_norm = norm[0].detach()
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

    def policy_action_async(self, instruction):
        """추론 지연 모사: 요청 시점의 관측으로 계산한 chunk 가 L 스텝 뒤 도착. 그동안 이전 계획을 실행하고,
        계획이 바닥나면 제자리 유지. 도착하면 이미 지나간 앞 L 개를 버리고 이어서 사용."""
        L = self.a.latency_steps
        if self.pending is None and (self.exec_left <= 0 or len(self.plan) <= L):
            out, norm = self.infer_chunk(instruction, commit=False)
            self.pending = [out, norm, L]
        if self.pending is not None and self.pending[2] <= 0:
            out, norm, _ = self.pending
            self.plan, self.plan_norm = out[L:], norm[L:]
            self.exec_left = self.n_act
            self.pending = None
        if len(self.plan) == 0:
            act = np.zeros(7, dtype=np.float32)
            act[6] = self.last_grip  # 새 계획이 올 때까지 제자리, 그리퍼 상태 유지
            self.hold_steps += 1
        else:
            act, self.plan = self.plan[0], self.plan[1:]
            if self.plan_norm is not None:
                self.plan_norm = self.plan_norm[1:]
        self.exec_left -= 1
        if self.pending is not None:
            self.pending[2] -= 1
        self.last_grip = float(act[6])
        return act

    def is_dithering(self):
        """최근 창 안에서 끝단이 좁은 영역에만 머물렀으면 idling."""
        w = self.a.dither_window
        if len(self.log["pos"]) < w:
            return False
        p = np.asarray(self.log["pos"][-w:]) * 100.0
        return float(np.linalg.norm(p.max(axis=0) - p.min(axis=0))) < self.a.dither_range_cm

    def escape_action(self):
        """흔들기 동작 한 개. 방향은 --dither-escape 에 따라 다르다."""
        mode, g = self.a.dither_escape, self.a.dither_gain
        grip = float(self.log["gq"][-1] < GRIPPER_OPEN_QPOS) * 2 - 1  # 지금 상태 유지
        if mode == "home":  # PIP 원본 — 초기 자세 쪽으로 (⛔ 제약 위반)
            d = self.home[0] - eef_pos(self.obs)
            v = np.clip(d / POS_SCALE, -1, 1) * g
        elif mode == "lift":
            v = np.array([0.0, 0.0, g])
        elif mode == "reverse":  # 최근 동작의 평균을 거꾸로 (짧게)
            k = min(self.a.dither_steps, len(self.acts))
            v = -np.mean([a[:3] for a in self.acts[-k:]], axis=0) if k > 0 else np.zeros(3)
            nv = float(np.linalg.norm(v))
            v = v / nv * g if nv > 1e-6 else np.array([0.0, 0.0, g])  # 움직임이 없었으면 위로
        else:  # random
            v = self.escape_rng.normal(size=3)
            v = v / max(np.linalg.norm(v), 1e-6) * g
        return np.concatenate([v, np.zeros(3), [grip]]).astype(np.float32)

    def maybe_escape(self):
        """흔들 때가 되었으면 흔들기 동작을 돌려준다. 아니면 None."""
        if self.a.dither_escape == "off" or not self.value_active:
            return None  # 전환 이후에만 적용 (value_active 가 전환 시점에 켜진다)
        if self.escape_left > 0:
            self.escape_left -= 1
            return self.escape_act
        if self.escape_cool > 0:
            self.escape_cool -= 1
            return None
        if self.is_dithering():
            self.escape_act = self.escape_action()
            self.escape_left = self.a.dither_steps - 1
            self.escape_cool = self.a.dither_cooldown
            self.escape_count += 1
            self.escape_at.append(len(self.log["pos"]))
            # 흔든 뒤에는 계획을 버리고 새로 뽑는다
            self.plan, self.exec_left, self.pending = np.zeros((0, 7)), 0, None
            return self.escape_act
        return None

    def policy_action(self, instruction):
        esc = self.maybe_escape()
        if esc is not None:
            return esc
        if self.a.latency_steps > 0:
            return self.policy_action_async(instruction)
        if self.exec_left <= 0 or len(self.plan) == 0:
            self.plan = self.infer_chunk(instruction)
            self.exec_left = self.n_act
        act, self.plan = self.plan[0], self.plan[1:]
        if self.plan_norm is not None:
            self.plan_norm = self.plan_norm[1:]
        self.exec_left -= 1
        return self.latch_grip(act)

    def latch_grip(self, act):
        """전환 직후 **놓지 말고 계속 쥐게** 한다.

        왜 (2026-09-23 측정)
          "그릇을 서랍에 넣어라" 를 **그릇을 쥔 채** 받아도 정책은 그릇을 놓는다 (10/10).
          B 가 바로 그 물체를 요구하는데도 놓는다 → **'지시가 바뀌면 일단 놓는다'가 학습된 반사**다.
          B 의 목표가 물체인 쌍은 전환 성공 1.4%, 가구·기구인 쌍은 44.6% 로 30배 차이가 난다.

        무엇을 하나
          전환 이후 N 스텝 동안 동작의 그리퍼 차원만 **닫힘(+1)** 으로 덮어쓴다.
          팔의 움직임은 정책이 낸 그대로다 → 초기 자세 복귀도, 스크립트 동작도 없다.
        """
        k = self.a.grip_latch
        if k <= 0 or not self.value_active or self.latch_left <= 0:
            return act
        self.latch_left -= 1
        act = np.asarray(act, dtype=np.float32).copy()
        act[6] = 1.0  # +1 = 닫기 (열기는 -1)
        return act

    RETREAT_MODES = {"retreat": (True, True), "release": (False, False),
                     "ret_pos": (True, False), "ret_rot": (False, True)}

    def apply_strategy(self, new_instruction, rec, key):
        s = self.a.strategy
        self.value_active = True
        if key == "to_b" and self.a.grip_latch > 0 and gripper_closed(self.obs):
            self.latch_left = self.a.grip_latch  # 쥐고 있을 때만 건다
        if self.a.switch_n_action_steps > 0:
            self.n_act = self.a.switch_n_action_steps  # 전환 이후로는 더 자주 재추론
        if self.a.switch_noise_seed >= 0:
            torch.manual_seed(self.a.switch_noise_seed)  # 전환 이후 노이즈 순서를 고정
        if self.a.latency_steps > 0:
            self.pending = None  # 이전 지시로 보낸 요청은 버림
            if s == "keep":
                self.exec_left = 0  # 지연 모사에서 keep = 새 요청을 바로 보내고, 도착할 때까지 이전 계획 계속
        if s == "finish_a":
            # 사람처럼 "하던 것만 마치고" 넘어간다. 즉시성을 포기하는 대신 자세가 정상으로 돌아온다.
            # A2(재개) 단계에서는 이미 A 가 끝나 있으므로 아무것도 하지 않는다.
            if key == "to_b" and not self.r.chk_a(self.env):
                why, n_fin = self.run_policy(self.r.chk_a.language, "A", self.a.finish_a_max, self.r.chk_a)
                rec["finish_a_steps"] = n_fin
                rec["finish_a_done"] = why == "success"
            self.plan, self.exec_left = np.zeros((0, 7)), 0
        elif s == "ret_part":
            rec[f"{key}_retreat_steps"] = self.retreat_partial(self.a.retreat_frac)
            self.plan, self.exec_left = np.zeros((0, 7)), 0
        elif s == "rollback":
            # 물체를 쥐고 있을 때만 되돌린다. 빈손이면 되돌릴 것이 없고,
            # 재개(A2) 단계에서 잘못 실행하면 그리퍼를 닫은 채 B 궤적을 되짚어 B 를 망친다.
            rec[f"{key}_rollback_steps"] = self.rollback() if gripper_closed(self.obs) else 0
            self.plan, self.exec_left = np.zeros((0, 7)), 0
        elif s in ("flush",) or s in self.RETREAT_MODES:
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
            self.exec_left = self.n_act

    # ---------- 환경 ----------
    def step(self, action, phase):
        # 수집 훅: 스크립트 구간(R)까지 포함해 (관측, 동작)을 기록하려면 이걸 쓴다.
        # 그래야 "스크립트로 만든 좋은 동작"을 정책에 증류할 수 있다.
        if self.record is not None:
            self.record(self.obs, action, phase)
        raw, _, _, _ = self.env._env.step(np.asarray(action, dtype=np.float64))
        self.obs = self.env._format_raw_obs(raw)
        self.acts.append(np.asarray(action, dtype=np.float32))
        closed = gripper_closed(self.obs)
        if closed and not self.was_closed:
            self.grasp_t = len(self.acts) - 1  # 가장 최근에 집은 시점 (rollback 범위의 기준)
        self.was_closed = closed
        self.log["pos"].append(eef_pos(self.obs))
        # 손목 회전도 남긴다 (초기 자세 대비 각도 분석용 — A 재개 실패 진단에 필요)
        self.log["rot"].append(
            Rotation.from_matrix(self.obs["robot_state"]["eef"]["mat"]).as_rotvec().astype(np.float32))
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

    def object_layout(self):
        """장면 안 모든 물체의 위치 (m). 놓은 물체가 B 를 방해하는지 확인용."""
        return {n_: [round(float(x), 4) for x in self.obj_pos(n_)] for n_ in self.inner.objects_dict}

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

    def rollback(self):
        """최근 동작을 거꾸로 재생해서 집었던 자리에 물체를 내려놓는다 (SwitchVLA 의 rollback).

        `retreat` 과 결정적으로 다른 점: **초기 자세로 가지 않는다.**
        되돌리는 구간을 "그리퍼가 닫힌 시점보다 rollback_pre 스텝 앞"까지로 제한하고,
        전체 길이도 rollback_max 로 묶는다. 그래서 끝나는 자리는 물체를 집었던 그 자리 근처다.

        동작이 끝단 delta(위치 3 + 회전 3 + 그리퍼)이므로, 순서를 뒤집고 부호를 바꾸면 되돌아간다.
        그리퍼는 "집기 시점"을 지나기 전까지 닫아두고, 그 시점부터 연다 (물체를 원래 자리에 놓기 위해).
        """
        n0 = len(self.log["pos"])
        hi = len(self.acts)
        if self.grasp_t is None or hi - self.grasp_t > self.a.rollback_max:
            return 0  # 집은 시점이 되돌릴 범위 밖 → 되돌리지 않는다
        g = self.grasp_t
        lo = max(0, g - self.a.rollback_pre, hi - self.a.rollback_max)
        opened = False
        for i in range(hi - 1, lo - 1, -1):
            a = self.acts[i]
            if i <= g:
                opened = True  # 집었던 시점을 지나면 그리퍼를 연다 = 물체를 그 자리에 놓음
            self.step(np.concatenate([-a[:6], [-1.0 if opened else 1.0]]), "R")
        for _ in range(10):  # 그리퍼가 실제로 열리고 물체가 놓일 시간
            self.step(np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32), "R")
        return len(self.log["pos"]) - n0

    def retreat_partial(self, frac):
        """쥔 물체를 내려놓고, 초기 자세 쪽으로 **frac 만큼만** 이동한다.

        frac = 1.0 이면 retreat 과 같고, 0.0 이면 release 와 같다.
        "얼마나 가까이 가야 정책이 다시 작동하는가"를 재기 위한 것.
        """
        home_pos, home_mat = self.home
        n0 = len(self.log["pos"])
        if gripper_closed(self.obs):
            zs = []
            for _ in range(40):  # 접촉으로 하강이 멈출 때까지
                self.step([0, 0, -0.3, 0, 0, 0, 1], "R")
                zs.append(eef_pos(self.obs)[2])
                if len(zs) >= 5 and zs[-5] - zs[-1] < 0.003:
                    break
        for _ in range(15):
            self.step([0, 0, 0, 0, 0, 0, -1], "R")
        if frac <= 0:
            for _ in range(5):
                self.step(get_libero_dummy_action(), "R")
            return len(self.log["pos"]) - n0
        # 목표 자세: 현재와 초기 사이를 frac 으로 보간
        cur_pos, cur_mat = eef_pos(self.obs).copy(), self.obs["robot_state"]["eef"]["mat"].copy()
        tgt_pos = cur_pos + frac * (home_pos - cur_pos)
        rel = Rotation.from_matrix(home_mat @ cur_mat.T).as_rotvec()
        tgt_mat = Rotation.from_rotvec(rel * frac).as_matrix() @ cur_mat
        for _ in range(30):  # 물체와 부딪히지 않게 먼저 들어올림
            if eef_pos(self.obs)[2] >= tgt_pos[2] - 0.01:
                break
            self.step([0, 0, 0.6, 0, 0, 0, -1], "R")
        for _ in range(100):
            dp = tgt_pos - eef_pos(self.obs)
            rot_err = Rotation.from_matrix(tgt_mat @ self.obs["robot_state"]["eef"]["mat"].T).as_rotvec()
            if np.linalg.norm(dp) < 0.01 and np.linalg.norm(rot_err) < 0.05:
                break
            self.step(np.concatenate([np.clip(dp / POS_SCALE, -1, 1),
                                      np.clip(rot_err / ROT_SCALE, -1, 1), [-1]]), "R")
        for _ in range(5):
            self.step(get_libero_dummy_action(), "R")
        return len(self.log["pos"]) - n0

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
        if getattr(args, "strategy", None) in ("rtc", "flush_rtc", "rtc_all"):
            from lerobot.policies.rtc.configuration_rtc import RTCConfig
            self.policy.config.rtc_config = RTCConfig(enabled=True, execution_horizon=args.rtc_horizon,
                                                      max_guidance_weight=args.rtc_guidance)
            self.policy.init_rtc_processor()
        self.policy.to(self.device).eval()
        self.last_prefix_feat = None
        self.noise_shift = None
        if getattr(args, "noise_shift", None):
            v = torch.as_tensor(np.load(args.noise_shift), dtype=torch.float32)
            if v.ndim == 1:  # 길이 32 → 모든 시점에 같은 이동
                v = v.unsqueeze(0).expand(self.policy.config.chunk_size, -1)
            self.noise_shift = (v * args.noise_shift_scale).to(self.device)
            print(f"노이즈 평균 이동: {args.noise_shift} × {args.noise_shift_scale} "
                  f"(노름 {float(self.noise_shift.norm()):.3f})", flush=True)
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
        lat = f"_lat{a.latency_steps}" if getattr(a, "latency_steps", 0) > 0 else ""
        cf = f"_cfg{a.cfg_w:g}" if getattr(a, "cfg_w", 1.0) != 1.0 else ""
        if getattr(a, "instr_repeat", 1) > 1:
            cf += f"_rep{a.instr_repeat}"
        if getattr(a, "b_text", None):
            cf += "_btext"
        if getattr(a, "grip_latch", 0) > 0:
            cf += f"_gl{a.grip_latch}"
        if cf and getattr(a, "cfg_from", "switch") == "always":
            cf += "a"
        sn = f"_sn{a.switch_n_action_steps}" if getattr(a, "switch_n_action_steps", 0) > 0 else ""
        ns = f"_ns{a.switch_noise_seed}" if getattr(a, "switch_noise_seed", -1) >= 0 else ""
        if getattr(a, "switch_noise_reseed", False):
            ns += "r"
        if getattr(a, "switch_noise_first_only", False):
            ns += "f"
        if getattr(a, "noise_seed_from_start", False):
            ns += "s"
        if getattr(a, "noise_shift", None):
            ns += f"_sh{pathlib.Path(a.noise_shift).stem}"
        if getattr(a, "noise_scale", 1.0) != 1.0:
            ns += f"_sc{a.noise_scale:g}"
        de = f"_de{a.dither_escape}" if getattr(a, "dither_escape", "off") != "off" else ""
        rf = f"_rf{a.retreat_frac:g}" if getattr(a, "strategy", "") == "ret_part" else ""
        sn = sn + ns + de + rf + cf
        return f"A{a.task_a}_{b}_{a.switch_at.replace(':', '')}_{a.strategy}{lat}{sn}"

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
        layout_at_switch = ep.object_layout()
        rec["objects_at_switch"] = layout_at_switch
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

        home_before = float(np.linalg.norm(eef_pos(ep.obs) - ep.home[0]))
        # --b-text 를 주면 그 글자를 넣는다. 성공 판정은 chk_b 그대로라
        # "지시를 못 알아들었는데도 B 가 이뤄지는가" 까지 같이 보인다.
        b_lang = a.b_text if a.b_text else self.chk_b.language
        rec["b_instruction_given"] = b_lang
        ep.apply_strategy(b_lang, rec, "to_b")
        # 초기 자세로 돌아갔는지 확인 (제약: 돌아가면 안 됨)
        rec["home_dist_before_cm"] = round(100 * home_before, 1)
        rec["home_dist_after_cm"] = round(100 * float(np.linalg.norm(eef_pos(ep.obs) - ep.home[0])), 1)
        why, n_b = ep.run_policy(b_lang, "B", a.max_steps, self.chk_b, watch_fn=watch_b)
        b_end = len(ep.log["pos"])
        rec.update(b_success=why == "success", b_steps=n_b, a_completed_during_b=a_during_b["v"],
                   escape_count=ep.escape_count, escape_at=list(ep.escape_at))
        rec.update(switch_metrics(ep.log, ts, b_end, ep.held))
        # 물체 배치: 전환 시점과 B 끝. 놓은 물체가 B 를 방해하는지 확인용
        rec["objects_at_b_end"] = ep.object_layout()
        if ep.held:
            rec["held_obj_moved_cm"] = round(100 * float(np.linalg.norm(
                np.array(rec["objects_at_b_end"][ep.held]) - np.array(layout_at_switch[ep.held]))), 1)

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
        if self.args.latency_steps > 0:
            rec["latency_steps"] = self.args.latency_steps
            rec["hold_steps"] = ep.hold_steps
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
                            held_z=np.asarray(ep.log["held_z"]),
                            eef_rotvec=np.asarray(ep.log["rot"], dtype=np.float32),
                            home_rotvec=Rotation.from_matrix(ep.home[1]).as_rotvec().astype(np.float32),
                            home_pos=np.asarray(ep.home[0], dtype=np.float32))
        if ep.frames:
            (out / "videos").mkdir(parents=True, exist_ok=True)
            write_video(str(out / "videos" / f"{self.tag()}_ep{ep_idx}.mp4"), np.stack(ep.frames), fps=30)
        ep.env.close()
        return rec


def summarize(recs):
    sw = [r for r in recs if r.get("switched")]
    out = {"n": len(recs), "n_switched": len(sw)}
    # 전환이 한 번도 없는 실행(태스크 단독 평가)에서는 전체 에피소드로 집계한다
    base = sw if sw else recs
    for k in ["holding_at_switch", "a_success", "b_success", "a_resume_success", "both_success", "drop"]:
        vals = [r[k] for r in base if r.get(k) is not None]
        if vals:
            out[k + "_rate"] = round(float(np.mean(vals)), 3)
    for k in ["jerk"]:
        vals = [r[k] for r in base if r.get(k) is not None]
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
