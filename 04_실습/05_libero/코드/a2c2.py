#!/usr/bin/env python
"""
A2C2 — 매 스텝 보정 네트워크 (2026-10-02)

출처
  "Leave No Observation Behind: Real-time Correction for VLA Action Chunks", arXiv 2509.23224 (2025)
  비교: "Understanding Asynchronous Inference Methods for VLA", arXiv 2605.08168 (2026) — SmolVLA·LIBERO,
        지연 8스텝에서 아무것도 안 함 25%, 학습 때 RTC 51%, A2C2 66%

왜
  SmolVLA 는 한 번 계산(0.56초)으로 50스텝 묶음을 만들고, 로봇은 그 묶음을 사진을 다시 보지 않고 따라간다.
  그사이 생긴 1~2cm 어긋남(그릇 테두리 빗나감)을 고칠 기회가 없다.
  작은 네트워크가 **매 스텝 최신 사진(정면·손목)** 을 보고, 묶음의 그 스텝 동작에 더할 작은 보정값을 낸다.
  SmolVLA 는 그대로 두고(얼림), 보정 네트워크만 학습한다. 1080 Ti 에서 스텝당 수 ms.

구성
  gen   : 학습 데이터의 시점 t0 마다 SmolVLA 로 묶음을 미리 계산해 둔다 (학습 때 지연 흉내내기 모델이면
          그 앞 L 스텝 정답을 고정해 넣는다 — 실제 로봇에서처럼)
  train : (t0+k 시점의 사진·상태, 묶음의 k번째 동작, k) → (선생 동작 − 묶음 동작) 를 맞힌다
  Corrector : 평가·로봇에서 매 스텝 부르는 객체

쓰는 법
  python a2c2.py gen   --policy outputs/v6c_model/merged --ttrtc --data data/v6_normal data/v6_switch --out data/a2c2_v6c
  python a2c2.py train --gen data/a2c2_v6c --out outputs/a2c2_v6c
"""
import argparse
import io
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

RES = 128          # 보정 네트워크가 보는 사진 크기
LOOK = 4           # 묶음에서 지금 동작과 그 뒤 몇 개를 같이 본다
K_MAX = 40         # 묶음 안 위치 k 를 이 범위에서 학습


# ───────────────────────── 모델 ─────────────────────────
def resnet18_gn():
    import torchvision
    m = torchvision.models.resnet18(weights=None, norm_layer=lambda c: nn.GroupNorm(max(1, c // 16), c))
    m.fc = nn.Identity()
    return m


def quat_to_axisangle(q):
    """(..., 4) xyzw → (..., 3). robosuite 와 같은 방식."""
    q = np.asarray(q, dtype=np.float64)
    w = np.clip(q[..., 3], -1.0, 1.0)
    den = np.sqrt(np.maximum(1.0 - w * w, 0.0))
    ang = 2.0 * np.arccos(w)
    out = np.where(den[..., None] > 1e-8, q[..., :3] * (ang / np.maximum(den, 1e-8))[..., None], 0.0)
    return out.astype(np.float32)


class A2C2Head(nn.Module):
    def __init__(self, tasks, d=256, layers=4):
        super().__init__()
        self.tasks = list(tasks)
        self.cnn = resnet18_gn()
        self.cam = nn.Parameter(torch.zeros(2, 512))
        self.img_proj = nn.Linear(512, d)
        self.state_proj = nn.Linear(8, d)
        self.act_proj = nn.Linear(7 * LOOK, d)
        self.time_proj = nn.Linear(3, d)
        self.task_emb = nn.Embedding(len(self.tasks) + 1, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 512), nn.GELU(), nn.Linear(512, 7))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)       # 처음에는 보정 0 (SmolVLA 그대로)
        self.register_buffer("state_mean", torch.zeros(8))
        self.register_buffer("state_std", torch.ones(8))
        self.register_buffer("act_mean", torch.zeros(7))
        self.register_buffer("act_std", torch.ones(7))

    def task_index(self, names):
        return torch.tensor([self.tasks.index(n) + 1 if n in self.tasks else 0 for n in names])

    def forward(self, img, wrist, state, acts, k, task):
        """img, wrist: (B,3,RES,RES) float 0~1 / state (B,8) 원래 단위 / acts (B,LOOK,7) 원래 단위 /
        k (B,) 묶음 안 위치 / task (B,) 정수. 반환: 정규화 공간 보정값 (B,7)."""
        B = img.shape[0]
        f = self.cnn(torch.cat([img, wrist], 0) - 0.5)
        f = f.view(2, B, 512) + self.cam[:, None, :]
        s = (state - self.state_mean) / self.state_std
        a = (acts - self.act_mean) / self.act_std
        kk = k.float()[:, None]
        tf = torch.cat([torch.sin(2 * np.pi * kk / 50), torch.cos(2 * np.pi * kk / 50), kk / 50], 1)
        toks = torch.stack([self.img_proj(f[0]), self.img_proj(f[1]), self.state_proj(s),
                            self.act_proj(a.reshape(B, -1)), self.time_proj(tf), self.task_emb(task)], 1)
        x = self.enc(torch.cat([self.cls.expand(B, -1, -1), toks], 1))
        return self.out(x[:, 0])


def prep_img(arr, rng=None):
    """HWC uint8 (256) → CHW float (RES). rng 가 있으면 위치 ±4픽셀, 밝기 ±10% 증강."""
    im = Image.fromarray(arr).resize((RES + 8, RES + 8) if rng is not None else (RES, RES), Image.BILINEAR)
    x = np.asarray(im, dtype=np.float32) / 255.0
    if rng is not None:
        i, j = rng.integers(0, 9, size=2)
        x = x[i:i + RES, j:j + RES] * rng.uniform(0.9, 1.1)
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def decode(blob):
    b = blob if isinstance(blob, (bytes, bytearray)) else bytes(blob)
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))


# ───────────────────────── 평가·로봇에서 쓰는 객체 ─────────────────────────
class Corrector:
    DIMS = {"all": list(range(7)), "posrot": list(range(6)), "pos": [0, 1, 2]}

    def __init__(self, path, device="cuda", scale=1.0, dims="all"):
        ck = torch.load(Path(path) / "head.pt", map_location="cpu")
        self.net = A2C2Head(ck["tasks"])
        self.net.load_state_dict(ck["state"])
        self.net.to(device).eval()
        self.device, self.scale = device, scale
        self.mask = np.zeros(7, dtype=np.float32)
        self.mask[self.DIMS[dims]] = 1.0          # 고칠 동작 차원 (10/2: 그리퍼 보정이 스토브 손잡이 쥐기를 방해하는지 확인용)

    @torch.no_grad()
    def __call__(self, obs, plan_raw, k, task):
        """obs: 환경 관측(dict), plan_raw: 지금부터 실행할 묶음 동작들 (n,7) 원래 단위, k: 지금 동작의 묶음 안 위치.
        반환: 보정을 더한 지금 동작 (7,)."""
        img = prep_img(np.asarray(obs["pixels"]["image"]))
        wr = prep_img(np.asarray(obs["pixels"]["image2"]))
        rs = obs["robot_state"]
        st = np.concatenate([rs["eef"]["pos"], quat_to_axisangle(rs["eef"]["quat"]), rs["gripper"]["qpos"]])
        acts = np.stack([plan_raw[min(i, len(plan_raw) - 1)] for i in range(LOOK)])
        t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32))[None].to(self.device)
        r = self.net(t(img), t(wr), t(st), t(acts), torch.tensor([k], device=self.device),
                     self.net.task_index([task]).to(self.device))[0]
        corr = (r * self.net.act_std).cpu().numpy() * self.scale * self.mask
        a = plan_raw[0] + corr
        a[:6] = np.clip(a[:6], -1, 1)
        a[6] = np.clip(a[6], -1, 1)
        return a.astype(np.float32)


# ───────────────────────── gen: SmolVLA 묶음 미리 계산 ─────────────────────────
def gen(a):
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.env_processor import LiberoProcessorStep
    from safetensors.torch import load_file
    import ttrtc
    if a.ttrtc:
        ttrtc.patch()
    dev = a.device
    pol = SmolVLAPolicy.from_pretrained(a.policy)
    pol.config.device = dev
    pol.to(dev).eval()
    pre, post = make_pre_post_processors(policy_cfg=pol.config, pretrained_path=a.policy,
                                         preprocessor_overrides={"device_processor": {"device": dev}})
    env_step = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
    st = load_file(str(Path(a.policy) / "policy_postprocessor_step_1_unnormalizer_processor.safetensors"))
    am, asd = st["action.mean"].numpy(), st["action.std"].numpy()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(f for d in a.data for f in Path(d).glob("episodes/*.npz"))
    rng = random.Random(a.seed)
    rng.shuffle(files)
    files = files[a.shard::a.nshard]
    if a.frac < 1:
        files = files[:int(len(files) * a.frac)]
    t_start, done = time.time(), 0
    for f in files:
        dst = out / f"{f.parent.parent.name}__{f.stem}.npz"
        if dst.exists():
            continue
        d = np.load(f, allow_pickle=True)
        n = len(d["action"])
        if n < a.L + 5:
            continue
        off = rng.randrange(a.stride)
        t0s = list(range(off, n - a.L - 2, a.stride))
        chunks = []
        for i in range(0, len(t0s), a.batch):
            ts = t0s[i:i + a.batch]
            obs = {"pixels": {"image": np.stack([decode(d["img"][t]) for t in ts]),
                              "image2": np.stack([decode(d["wrist"][t]) for t in ts])},
                   "robot_state": {"eef": {"pos": d["eef_pos"][ts], "quat": d["eef_quat"][ts]},
                                   "gripper": {"qpos": d["grip"][ts]}}}
            with torch.inference_mode():
                b = preprocess_observation(obs)
                b["task"] = [str(d["task"])] * len(ts)
                b = pre(env_step(b))
                if a.ttrtc:
                    pref = np.stack([d["action"][t:t + a.L] for t in ts])
                    ttrtc.STATE["prefix"] = torch.as_tensor((pref - am) / asd, dtype=torch.float32)
                norm = pol.predict_action_chunk(b)
                raw = post(norm.clone()).float().cpu().numpy()
            chunks.append(raw)
        np.savez_compressed(dst, t0=np.array(t0s), base=np.concatenate(chunks).astype(np.float16),
                            src=str(f), task=str(d["task"]), L=a.L if a.ttrtc else 0)
        done += 1
        if done % 20 == 0:
            el = time.time() - t_start
            print(f"{done}/{len(files)} 에피소드, {el / done:.1f}초/에피소드", flush=True)
    print(f"끝: {done} 에피소드, {time.time() - t_start:.0f}초", flush=True)


# ───────────────────────── train ─────────────────────────
class JpegStore:
    """JPEG 바이트들을 큰 uint8 배열 하나에 이어 붙여 둔다. 파이썬 객체가 없어 DataLoader 작업자들이
    메모리를 복사하지 않고 같이 쓴다 (31GB PC 에서 중요)."""

    def __init__(self):
        self.parts, self.offs, self.n = [], [0], 0

    def add(self, blobs):
        i0 = len(self.offs) - 1
        for b in blobs:
            b = b if isinstance(b, (bytes, bytearray)) else bytes(b)
            self.parts.append(b)
            self.n += len(b)
            self.offs.append(self.n)
        return i0

    def freeze(self):
        self.buf = np.frombuffer(b"".join(self.parts), dtype=np.uint8)
        self.offs = np.array(self.offs, dtype=np.int64)
        self.parts = None

    def get(self, i):
        return self.buf[self.offs[i]:self.offs[i + 1]].tobytes()


class HeadData(torch.utils.data.Dataset):
    def __init__(self, gen_files, k_min, k_max, aug):
        self.eps, self.index, self.aug = [], [], aug
        self.jpg = JpegStore()
        for g in gen_files:
            G = np.load(g, allow_pickle=True)
            d = np.load(str(G["src"]), allow_pickle=True)
            n = len(d["action"])
            ep = {"img": self.jpg.add(d["img"]), "wrist": self.jpg.add(d["wrist"]), "pos": d["eef_pos"],
                  "quat": d["eef_quat"], "grip": d["grip"], "action": d["action"], "task": str(d["task"]),
                  "t0": G["t0"], "base": G["base"].astype(np.float32)}
            self.eps.append(ep)
            e = len(self.eps) - 1
            for j, t0 in enumerate(G["t0"]):
                for k in range(k_min, k_max):
                    if t0 + k < n:
                        self.index.append((e, j, k))
        self.jpg.freeze()
        self.index = np.array(self.index, dtype=np.int32)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        e, j, k = self.index[i]
        ep = self.eps[e]
        t = int(ep["t0"][j]) + k
        rng = np.random.default_rng() if self.aug else None
        base = ep["base"][j]
        acts = np.stack([base[min(k + q, len(base) - 1)] for q in range(LOOK)])
        st = np.concatenate([ep["pos"][t], quat_to_axisangle(ep["quat"][t]), ep["grip"][t]]).astype(np.float32)
        img, wr = decode(self.jpg.get(ep["img"] + t)), decode(self.jpg.get(ep["wrist"] + t))
        return {"img": prep_img(img, rng), "wrist": prep_img(wr, rng),
                "state": st, "acts": acts.astype(np.float32), "k": k, "task": ep["task"],
                "target": (ep["action"][t] - base[k]).astype(np.float32)}


def collate(items):
    out = {k: torch.as_tensor(np.stack([x[k] for x in items])) for k in ("img", "wrist", "state", "acts", "target")}
    out["k"] = torch.tensor([x["k"] for x in items])
    out["task"] = [x["task"] for x in items]
    return out


def train(a):
    torch.manual_seed(a.seed)
    gfiles = sorted(Path(a.gen).glob("*.npz"))
    random.Random(a.seed).shuffle(gfiles)
    nv = max(1, int(len(gfiles) * 0.05))
    tr = HeadData(gfiles[nv:], a.k_min, a.k_max, aug=True)
    va = HeadData(gfiles[:nv], a.k_min, a.k_max, aug=False)
    tasks = sorted({ep["task"] for ep in tr.eps})
    allst = np.concatenate([np.concatenate([ep["pos"], quat_to_axisangle(ep["quat"]), ep["grip"]], 1) for ep in tr.eps])
    alla = np.concatenate([ep["action"] for ep in tr.eps])
    net = A2C2Head(tasks)
    dev = a.device
    net.state_mean.copy_(torch.as_tensor(allst.mean(0)))
    net.state_std.copy_(torch.as_tensor(allst.std(0) + 1e-3))
    net.act_mean.copy_(torch.as_tensor(alla.mean(0)))
    net.act_std.copy_(torch.as_tensor(alla.std(0) + 1e-3))
    net.to(dev)
    print(f"보정 네트워크 {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M, 학습 {len(tr)} / 검증 {len(va)} 샘플, "
          f"태스크 {len(tasks)}", flush=True)
    dl = torch.utils.data.DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                                     collate_fn=collate, drop_last=True, persistent_workers=a.workers > 0)
    vdl = torch.utils.data.DataLoader(va, batch_size=a.batch, shuffle=True, num_workers=min(a.workers, 2),
                                      collate_fn=collate)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.03)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    w = torch.tensor([1, 1, 1, 1, 1, 1, a.grip_w], device=dev, dtype=torch.float32)

    def run(b):
        task = net.task_index(b["task"]).to(dev)
        pred = net(b["img"].to(dev), b["wrist"].to(dev), b["state"].to(dev), b["acts"].to(dev), b["k"].to(dev), task)
        tgt = b["target"].to(dev) / net.act_std
        return ((pred - tgt) ** 2 * w).mean(), ((tgt) ** 2 * w).mean()

    step, t0, hist = 0, time.time(), []
    while step < a.steps:
        for b in dl:
            net.train()
            loss, _ = run(b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % a.eval_every == 0 or step == a.steps:
                net.eval()
                vl, v0, nb = 0.0, 0.0, 0
                with torch.no_grad():
                    for i, vb in enumerate(vdl):
                        if i >= 30:
                            break
                        l, l0 = run(vb)
                        vl, v0, nb = vl + l.item(), v0 + l0.item(), nb + 1
                rec = {"step": step, "loss": round(loss.item(), 4), "val": round(vl / nb, 4),
                       "val_no_corr": round(v0 / nb, 4), "sec": round(time.time() - t0)}
                print(json.dumps(rec), flush=True)
                hist.append(rec)
                torch.save({"tasks": tasks, "state": net.state_dict(), "args": vars(a)}, out / "head.pt")
                (out / "history.json").write_text(json.dumps(hist, indent=1))
            if step >= a.steps:
                break
    print("저장:", out / "head.pt", flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--policy", required=True)
    g.add_argument("--data", nargs="+", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--ttrtc", action="store_true")
    g.add_argument("--L", type=int, default=11, help="고정해 넣는 앞부분 길이 (실제 지연)")
    g.add_argument("--stride", type=int, default=12, help="몇 스텝마다 묶음을 계산하나")
    g.add_argument("--batch", type=int, default=8)
    g.add_argument("--frac", type=float, default=1.0)
    g.add_argument("--shard", type=int, default=0)
    g.add_argument("--nshard", type=int, default=1)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", default="cuda")
    t = sub.add_parser("train")
    t.add_argument("--gen", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--k-min", type=int, default=0)
    t.add_argument("--k-max", type=int, default=K_MAX)
    t.add_argument("--steps", type=int, default=30000)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--grip-w", type=float, default=0.3)
    t.add_argument("--workers", type=int, default=6)
    t.add_argument("--eval-every", type=int, default=1000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda")
    a = p.parse_args()
    gen(a) if a.cmd == "gen" else train(a)


if __name__ == "__main__":
    main()
