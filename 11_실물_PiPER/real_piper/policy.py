"""VLA (SmolVLA + LoRA, w11m·w12m 등) 를 실물에서 부르기 — 시뮬 평가와 같은 입력·같은 지연 처리 (2026-11-04).

입력은 시뮬과 똑같이 만든다 (switch_experiment.Episode.infer_chunk 와 같은 길):
  obs = {"pixels": {"image": 위 카메라, "image2": 손목 카메라},          # 512×512 RGB, 위아래 뒤집힌 꼴 (cameras.py)
         "robot_state": {"eef": {"pos", "quat"(xyzw, 손 몸), "mat"(손끝)}, "gripper": {"qpos"}, "joints": {"pos"}}}
  → preprocess_observation → Runner.prep (사진 뒤집기·상태 8개 + 물체 위치 입력 24개) → policy → 동작 묶음.
지연: 시뮬 평가(--latency-steps L --ttrtc --n-action-steps 1)와 같은 규칙 (switch_experiment.Episode.policy_action_async).
  · 계획이 바닥나기 전(L 개 이하 남음)이나 한 동작을 쓰면 바로 다음 추론을 시작하고, 그동안 실행할 앞 동작(최대 L 개)을 접두로 넣는다
  · 'steps': 추론이 늘 L 스텝 걸린다고 본다 (시뮬 시험).  'thread': 다른 스레드에서 실제 시간으로 — 도착하면 그동안 실행한 만큼 버린다
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class VLA:
    def __init__(self, model_dir, latency_steps=9, cuda_graph=True, world_cfg="piper_sim/worlds/default.json",
                 detect_model=None, camera_calib=None):
        os.environ.setdefault("PIPER_NOPED", "1")
        os.environ.setdefault("PIPER_WORLD", str(ROOT / world_cfg) if not os.path.isabs(world_cfg) else world_cfg)
        import torch
        import piper_sim.piper_robot as pr
        pr.use_piper_in_lerobot()
        import piper_sim.world as W
        W.install()
        import switch_experiment as sx
        import piper_sim.world_eval as WEv
        self.torch, self.sx, self.W = torch, sx, W
        argv = ["--task-a", "0", "--policy", str(model_dir), "--latency-steps", str(int(latency_steps)), "--ttrtc", "--n-action-steps", "1"]
        if cuda_graph:
            argv.append("--cuda-graph")
        args = sx.fill_defaults(sx.build_parser().parse_args(argv))
        args.task_a, args.task_b, args.strategy = 0, None, "none"
        self.r = sx.Runner(args)
        WEv.pad_lang(self.r)
        if camera_calib is not None:                       # 위 카메라 보정값 (calib_top_camera.py) — 물체 찾기가 쓴다
            W.CAMERA_CALIB = (list(camera_calib["pos"]), list(camera_calib["quat_wxyz"]))
        if detect_model and getattr(self.r, "wf", None) is not None:
            import piper_sim.world_detect as WD          # 실물 사진으로 맞춘 색 분포 (fit_colors.py)
            self.r.wf.det = WD.Detector(path=str(detect_model))
        self.abs_pos = bool(getattr(self.r, "abs_pos", False))
        self.reset()

    def reset(self):
        if getattr(self.r, "wf", None) is not None:
            self.r.wf.reset()                              # 물체 위치 기억을 비운다 (시뮬은 장면마다)

    def infer(self, obs, instruction, prefix=None):
        """(동작 묶음 (T, 7) numpy, 정규화 묶음 (T, 7) torch)."""
        import ttrtc
        from lerobot.envs.utils import preprocess_observation
        r = self.r
        if getattr(r, "canon", None) is not None:
            instruction = r.canon(instruction)
        with self.torch.inference_mode():
            if prefix is not None and len(prefix) > 0:
                ttrtc.STATE["prefix"] = prefix
            batch = preprocess_observation(self.sx.add_batch_dim(obs))
            batch["task"] = [instruction]
            batch = r.prep(batch, obs)
            norm = r.policy.predict_action_chunk(batch)
            chunk = r.post(norm.clone())
        return chunk[0].float().cpu().numpy(), norm[0].detach()


def make_obs(top, wrist, kin, q, grip_total_m, q_offset=0.0):
    """시뮬 관측과 같은 꼴. grip_total_m = 두 손가락 사이 (m). 시뮬 손가락 값은 [반, −반] (활짝 0.035)."""
    pos, mat, quat = kin.fk(q)
    half = float(np.clip(grip_total_m / 2 + q_offset, 0.0, 0.035))
    return {"pixels": {"image": top, "image2": wrist},
            "robot_state": {"eef": {"pos": pos, "quat": quat, "mat": mat},
                            "gripper": {"qpos": np.array([half, -half]), "qvel": np.zeros(2)},
                            "joints": {"pos": np.asarray(q, dtype=np.float64)[:6].copy(), "vel": np.zeros(6)}}}


class Planner:
    def __init__(self, vla, latency_steps=9, mode="thread"):
        self.vla, self.L, self.mode = vla, int(latency_steps), mode
        self.pool = ThreadPoolExecutor(max_workers=1) if mode == "thread" else None
        self.reset()

    def reset(self):
        self.plan, self.plan_norm, self.pending = np.zeros((0, 7), np.float32), None, None
        self.exec_left, self.last_grip, self.n_infer = 0, -1.0, 0

    def switch(self):
        """지시가 바뀌면: 시뮬 평가처럼 이어서 (계획을 버리지 않음 — 다음 추론부터 새 지시)."""

    def next(self, obs, instruction, eef_pos):
        L = self.L
        if self.pending is None and (self.exec_left <= 0 or len(self.plan) <= L):
            k = min(L, len(self.plan))
            prefix = self.plan_norm[:k] if (self.plan_norm is not None and k > 0) else None
            if self.mode == "steps":
                out, norm = self.vla.infer(obs, instruction, prefix)
                self.pending = {"res": (out, norm), "left": L, "k": k}
            else:
                self.pending = {"fut": self.pool.submit(self.vla.infer, obs, instruction, prefix), "k": k, "done_old": 0}
        if self.pending is not None:
            arrived = (self.pending["left"] <= 0) if self.mode == "steps" else self.pending["fut"].done()
            if arrived:
                out, norm = self.pending["res"] if self.mode == "steps" else self.pending["fut"].result()
                drop = self.pending["k"] if self.mode == "steps" else self.pending["done_old"]
                self.plan, self.plan_norm = out[drop:], norm[drop:]
                self.exec_left, self.pending = 1, None
                self.n_infer += 1
        if len(self.plan) == 0:
            act = np.zeros(7, dtype=np.float32)
            act[6] = self.last_grip                        # 새 계획이 올 때까지 제자리, 그리퍼 그대로
            if self.vla.abs_pos:
                act[:3] = eef_pos
        else:
            act = np.array(self.plan[0], dtype=np.float32)
            self.plan = self.plan[1:]
            if self.plan_norm is not None:
                self.plan_norm = self.plan_norm[1:]
            if self.pending is not None and self.mode == "thread":
                self.pending["done_old"] += 1
        self.exec_left -= 1
        if self.pending is not None and self.mode == "steps":
            self.pending["left"] -= 1
        self.last_grip = float(act[6])
        return act


def to_delta(act, eef_pos, abs_pos=True):
    """절대 목표 위치 모델: 위치 3개를 '지금 손끝과의 차이 / 0.05' 로 (switch_experiment.Episode.to_delta)."""
    a = np.array(act, dtype=np.float32, copy=True)
    if abs_pos:
        a[:3] = np.clip((a[:3] - eef_pos) / 0.05, -1, 1)
    return a
