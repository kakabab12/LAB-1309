#!/usr/bin/env python
"""
SmolVLA LoRA 파인튜닝 (연구실 GTX 1080 Ti, fp32).

목적: 어떤 자세에서 시작하든 태스크를 수행하게 만들어, **전환할 때 초기 자세로 되돌리지 않아도**
      되게 한다. (연구주제.md 제약: 전환 시 리셋 금지)

데이터: collect_robust_data.py 가 모은 "자세를 틀어놓고도 성공한" 궤적 (self-imitation)

LoRA 대상: action expert(lm_expert)의 q/k/v/o_proj, gate/up/down_proj
          VLM 과 비전 인코더는 체크포인트 설정(train_expert_only, freeze_vision_encoder)대로 동결

예시:
  python train_lora.py --data data/robust --steps 3000 --batch-size 2 --out outputs/lora_v1
"""

from __future__ import annotations

import argparse
import io
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.env_processor import LiberoProcessorStep

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="HuggingFaceVLA/smolvla_libero")
    p.add_argument("--data", nargs="+", default=["data/robust"], help="데이터 폴더 여러 개 가능")
    p.add_argument("--balance", action="store_true",
                   help="지시문(태스크)별로 같은 비중이 되게 샘플링 — 한 태스크 데이터가 몰려 다른 걸 잊는 문제 완화")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--out", default="outputs/lora_v1")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def decode(blob) -> np.ndarray:
    """저장된 JPEG 바이트 → HWC uint8 배열."""
    b = blob if isinstance(blob, (bytes, bytearray)) else bytes(blob)
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))


class ChunkDataset(Dataset):
    """(에피소드, 시점) → 이미지 2장 + 로봇 상태 + 앞으로 chunk_size 스텝의 동작."""

    def __init__(self, files, chunk_size):
        self.chunk = chunk_size
        self.eps, self.index = [], []
        for f in files:
            d = np.load(f, allow_pickle=True)
            ep = {k: d[k] for k in ["img", "wrist", "eef_pos", "eef_quat", "grip", "action"]}
            ep["task"] = str(d["task"])
            self.eps.append(ep)
            self.index += [(len(self.eps) - 1, t) for t in range(len(ep["action"]))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        e, t = self.index[i]
        ep = self.eps[e]
        n = len(ep["action"])
        idx = np.clip(np.arange(t, t + self.chunk), 0, n - 1)
        actions = ep["action"][idx]
        is_pad = np.arange(t, t + self.chunk) >= n
        return {
            "img": decode(ep["img"][t]),
            "wrist": decode(ep["wrist"][t]),
            "eef_pos": ep["eef_pos"][t], "eef_quat": ep["eef_quat"][t], "grip": ep["grip"][t],
            "action": actions.astype(np.float32), "is_pad": is_pad, "task": ep["task"],
        }


def collate(items):
    return {
        "pixels": {"image": np.stack([x["img"] for x in items]),
                   "image2": np.stack([x["wrist"] for x in items])},
        "robot_state": {"eef": {"pos": np.stack([x["eef_pos"] for x in items]),
                                "quat": np.stack([x["eef_quat"] for x in items])},
                        "gripper": {"qpos": np.stack([x["grip"] for x in items])}},
        "action": torch.from_numpy(np.stack([x["action"] for x in items])),
        "action_is_pad": torch.from_numpy(np.stack([x["is_pad"] for x in items])),
        "task": [x["task"] for x in items],
    }


class Trainer:
    def __init__(self, args):
        self.a = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = SmolVLAPolicy.from_pretrained(args.policy)
        self.policy.config.device = self.device
        self.policy.to(self.device)
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.policy.config, pretrained_path=args.policy,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )
        self.env_step = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
        self.chunk_size = self.policy.config.chunk_size
        self.apply_lora()

    def apply_lora(self):
        """action expert(LlamaModel)에만 LoRA 를 붙이고 나머지는 모두 동결."""
        from peft import LoraConfig, get_peft_model
        for prm in self.policy.parameters():
            prm.requires_grad_(False)
        expert = self.policy.model.vlm_with_expert.lm_expert
        cfg = LoraConfig(r=self.a.lora_r, lora_alpha=self.a.lora_alpha, lora_dropout=self.a.lora_dropout,
                         bias="none", target_modules=LORA_TARGETS)
        self.lora_expert = get_peft_model(expert, cfg)
        self.policy.model.vlm_with_expert.lm_expert = self.lora_expert
        tr = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in self.policy.parameters())
        print(f"LoRA 적용: r={self.a.lora_r}, 학습 파라미터 {tr / 1e6:.2f}M / 전체 {tot / 1e6:.1f}M "
              f"({100 * tr / tot:.2f}%)", flush=True)

    def batch_to_model(self, batch):
        b = preprocess_observation({"pixels": batch["pixels"], "robot_state": batch["robot_state"]})
        b["task"] = batch["task"]
        b["action"] = batch["action"]
        b = self.env_step(b)
        b = self.pre(b)
        b["actions_id_pad"] = batch["action_is_pad"].to(self.device)  # forward 가 읽는 키 이름
        return b

    def loss(self, batch):
        loss, _ = self.policy(self.batch_to_model(batch))  # (loss, loss_dict)
        return loss

    def run(self):
        a = self.a
        torch.manual_seed(a.seed)
        random.seed(a.seed)
        files = sorted(f for d in a.data for f in Path(d).glob("episodes/*.npz"))
        if not files:
            raise SystemExit(f"{a.data}/episodes 에 데이터가 없습니다. collect_robust_data.py 를 먼저 돌리세요.")
        random.shuffle(files)
        n_val = max(1, int(len(files) * a.val_frac))
        val_files, train_files = files[:n_val], files[n_val:]
        chunk = self.chunk_size
        train_ds, val_ds = ChunkDataset(train_files, chunk), ChunkDataset(val_files, chunk)
        print(f"에피소드 학습 {len(train_files)} / 검증 {len(val_files)}, "
              f"프레임 {len(train_ds)} / {len(val_ds)}", flush=True)
        sampler = None
        if a.balance:
            from collections import Counter
            from torch.utils.data import WeightedRandomSampler
            tasks = [train_ds.eps[e]["task"] for e, _ in train_ds.index]
            cnt = Counter(tasks)
            print("태스크별 프레임:", dict(cnt), flush=True)
            weights = torch.tensor([1.0 / cnt[x] for x in tasks], dtype=torch.double)
            sampler = WeightedRandomSampler(weights, num_samples=len(tasks), replacement=True)
        dl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=sampler is None, sampler=sampler,
                        collate_fn=collate, num_workers=2, drop_last=True, persistent_workers=True)
        vdl = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False, collate_fn=collate, num_workers=1)

        opt = torch.optim.AdamW([p for p in self.policy.parameters() if p.requires_grad], lr=a.lr,
                                weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        hist = []
        step, t0, acc_loss = 0, time.time(), 0.0
        self.policy.train()
        while step < a.steps:
            for batch in dl:
                loss = self.loss(batch) / a.grad_accum
                loss.backward()
                acc_loss += loss.item() * a.grad_accum
                if (step + 1) % a.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_([p for p in self.policy.parameters() if p.requires_grad], 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                step += 1
                sched.step()
                if step % a.log_every == 0:
                    mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                    rec = {"step": step, "loss": round(acc_loss / a.log_every, 5),
                           "lr": round(sched.get_last_lr()[0], 7), "vram_GB": round(mem, 2),
                           "sec": round(time.time() - t0, 1)}
                    print(json.dumps(rec), flush=True)
                    hist.append(rec)
                    acc_loss = 0.0
                if step % a.eval_every == 0:
                    v = self.validate(vdl)
                    print(json.dumps({"step": step, "val_loss": round(v, 5)}), flush=True)
                    hist.append({"step": step, "val_loss": round(v, 5)})
                if step % a.save_every == 0 or step == a.steps:
                    self.save(out, step, hist)
                if step >= a.steps:
                    break
        self.save(out, step, hist)
        self.export_merged(out / f"step_{step}", out / "merged")

    @torch.no_grad()
    def validate(self, vdl, max_batches=40):
        self.policy.eval()
        tot, n = 0.0, 0
        for i, batch in enumerate(vdl):
            if i >= max_batches:
                break
            tot += self.loss(batch).item()
            n += 1
        self.policy.train()
        return tot / max(n, 1)

    def save(self, out, step, hist):
        """학습 중에는 LoRA 어댑터만 저장 (가볍고, 옵티마이저 상태를 건드리지 않음)."""
        d = out / f"step_{step}"
        d.mkdir(parents=True, exist_ok=True)
        self.lora_expert.save_pretrained(d)  # adapter_model.safetensors + adapter_config.json
        (out / "history.json").write_text(json.dumps(hist, ensure_ascii=False, indent=2))
        print(f"어댑터 저장: {d}", flush=True)

    def export_merged(self, adapter_dir, dest):
        """LoRA 를 본체에 합쳐 SmolVLAPolicy.from_pretrained 로 바로 쓸 수 있게 저장."""
        from peft import PeftModel
        base = SmolVLAPolicy.from_pretrained(self.a.policy)
        expert = PeftModel.from_pretrained(base.model.vlm_with_expert.lm_expert, str(adapter_dir))
        base.model.vlm_with_expert.lm_expert = expert.merge_and_unload()
        merged = base
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(dest)
        self.pre.save_pretrained(dest)
        self.post.save_pretrained(dest)
        print(f"합친 모델 저장: {dest}  (평가: --policy {dest})", flush=True)
        return dest


if __name__ == "__main__":
    Trainer(parse_args()).run()
