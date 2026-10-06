"""Train one LeRobot (v0.3.3) ACT policy per stacking stage on the scripted demonstrations.

Same policy class / defaults as the real SO-101 system (ResNet18, chunk 100, CVAE, L1 + KL),
inputs: observation.state (6) + observation.images.{front,top} (3x120x160), output: action (6).

usage: python train_act.py --stage 1 --episodes 50 --steps 80000 --out runs/stage1
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

CAMS = ("front", "top")


class Demos:
    """Demonstrations from JPEG packs (convert_jpeg.py), held entirely in RAM.

    Only the bytes of the episodes actually used are kept, so n=100 from a 200-episode pack costs half.
    """

    def __init__(self, paths: list[Path], n: int):
        self.state, self.action, self.off, self.size, self.blobs, self.which = [], [], [], [], [], []
        need = n
        for path in paths:
            if need <= 0:
                break
            with np.load(path) as z:
                E = min(need, z["state"].shape[0])
                off, size = z["off"][:E], z["size"][:E]
                lo, hi = int(off.min()), int((off + size).max())
                self.blobs.append(np.array(z["blob"][lo:hi]))
                self.off.append(off - lo)
                self.size.append(size)
                self.state.append(z["state"][:E])
                self.action.append(z["action"][:E])
                self.which += [(len(self.blobs) - 1, e) for e in range(E)]
            need -= E
        assert len(self.which) == n, f"only {len(self.which)} episodes available, wanted {n}"
        self.T = self.state[0].shape[1]
        self.nbytes = sum(b.nbytes for b in self.blobs)

    def __len__(self) -> int:
        return len(self.which)

    def episode(self, i: int):
        p, e = self.which[i]
        return self.state[p][e], self.action[p][e]

    def image(self, i: int, t: int, cam: int) -> np.ndarray:
        p, e = self.which[i]
        o, n = self.off[p][e, t, cam], self.size[p][e, t, cam]
        bgr = cv2.imdecode(self.blobs[p][o : o + n], cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def compute_stats(d: Demos) -> dict:
    """Mean/std for state, action and each camera (images: every 15th frame of up to 40 episodes)."""
    S = np.concatenate([d.episode(i)[0] for i in range(len(d))])
    A = np.concatenate([d.episode(i)[1] for i in range(len(d))])
    stats = {
        "observation.state": {"mean": S.mean(0), "std": S.std(0) + 1e-3},
        "action": {"mean": A.mean(0), "std": A.std(0) + 1e-3},
    }
    for c, cam in enumerate(CAMS):
        s1, s2, n = np.zeros(3), np.zeros(3), 0
        for i in range(0, len(d), max(1, len(d) // 40)):
            for t in range(0, d.T, 15):
                x = d.image(i, t, c).reshape(-1, 3).astype(np.float64) / 255.0
                s1 += x.sum(0)
                s2 += (x * x).sum(0)
                n += len(x)
        mean = s1 / n
        std = np.sqrt(np.maximum(s2 / n - mean**2, 1e-8))
        stats[f"observation.images.{cam}"] = {"mean": mean.reshape(3, 1, 1), "std": std.reshape(3, 1, 1)}
    return {k: {s: torch.tensor(v, dtype=torch.float32) for s, v in d_.items()} for k, d_ in stats.items()}


def release_heap() -> None:
    """Hand freed heap pages back to the OS (glibc keeps them otherwise)."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def make_config(img_hw: tuple[int, int], device: str) -> ACTConfig:
    h, w = img_hw
    return ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, h, w)),
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, h, w)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        chunk_size=100,
        n_action_steps=100,
        device=device,
    )


class Sampler:
    """Uniform random (episode, t) samples with ACT action chunks and padding flags."""

    def __init__(self, demos: Demos, chunk: int, rng: np.random.Generator):
        self.d, self.chunk, self.rng = demos, chunk, rng

    def batch(self, bs: int, device: str) -> dict:
        T = self.d.T
        st, act, pad, imgs = [], [], [], {c: [] for c in CAMS}
        for _ in range(bs):
            i = int(self.rng.integers(len(self.d)))
            t = int(self.rng.integers(T))
            S, A = self.d.episode(i)
            idx = np.arange(t, t + self.chunk)
            pad.append(idx >= T)
            st.append(S[t])
            act.append(A[np.minimum(idx, T - 1)])
            for c, cam in enumerate(CAMS):
                imgs[cam].append(self.d.image(i, t, c))
        b = {
            "observation.state": torch.from_numpy(np.stack(st)),
            "action": torch.from_numpy(np.stack(act)),
            "action_is_pad": torch.from_numpy(np.stack(pad)),
        }
        for cam in CAMS:
            b[f"observation.images.{cam}"] = torch.from_numpy(np.stack(imgs[cam])).permute(0, 3, 1, 2).float() / 255.0
        return {k: v.to(device, non_blocking=True) for k, v in b.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--data", nargs="+", default=["data/stage{stage}.jpk.npz", "data/stage{stage}_more.jpk.npz"])
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    paths = [Path(p.format(stage=args.stage)) for p in args.data]
    demos = Demos([p for p in paths if p.exists()], args.episodes)
    img_hw = demos.image(0, 0, 0).shape[:2]
    stats = compute_stats(demos)
    cfg = make_config(img_hw, device)
    policy = ACTPolicy(cfg, dataset_stats=stats).to(device)
    policy.train()
    opt = torch.optim.AdamW(policy.get_optim_params(), lr=cfg.optimizer_lr, weight_decay=cfg.optimizer_weight_decay)
    sampler = Sampler(demos, cfg.chunk_size, rng)
    release_heap()
    n_params = sum(p.numel() for p in policy.parameters())
    meta = {"stage": args.stage, "episodes": len(demos), "frames": len(demos) * demos.T, "image_bytes": demos.nbytes,
            "data": [str(p) for p in paths if p.exists()], "steps": args.steps,
            "batch": args.batch, "lr": cfg.optimizer_lr, "params": n_params, "img_hw": list(img_hw),
            "chunk_size": cfg.chunk_size, "kl_weight": cfg.kl_weight, "device": torch.cuda.get_device_name(0)}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)

    log = []
    t0 = time.time()
    acc = {"loss": 0.0, "l1_loss": 0.0, "kld_loss": 0.0}
    for step in range(1, args.steps + 1):
        batch = sampler.batch(args.batch, device)
        loss, info = policy.forward(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        opt.step()
        acc["loss"] += loss.item()
        acc["l1_loss"] += info.get("l1_loss", 0.0)
        acc["kld_loss"] += info.get("kld_loss", 0.0)
        if step % 5000 == 0:
            release_heap()
        if step % 200 == 0:
            rec = {"step": step, **{k: v / 200 for k, v in acc.items()}, "elapsed_s": time.time() - t0}
            log.append(rec)
            acc = {k: 0.0 for k in acc}
            if step % 2000 == 0:
                print(json.dumps(rec), flush=True)
                (out / "train_log.json").write_text(json.dumps(log))
        if step % args.save_every == 0 or step == args.steps:
            policy.save_pretrained(out / f"ckpt_{step:06d}")
    (out / "train_log.json").write_text(json.dumps(log))
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
