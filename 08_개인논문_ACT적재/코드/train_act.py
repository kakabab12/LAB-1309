"""Train one LeRobot (v0.3.3) ACT policy per stacking stage on the scripted demonstrations.

Same policy class / defaults as the real SO-101 system (ResNet18, chunk 100, CVAE, L1 + KL),
inputs: observation.state (6) + observation.images.{front,top} (3x120x160), output: action (6).

usage: python train_act.py --stage 1 --episodes 50 --steps 80000 --out runs/stage1
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

CAMS = ("front", "top")


def load_episodes(paths: list[Path], n: int, images_in_ram: bool = False) -> list[dict]:
    """State/action go to RAM; images stay in the (lzf, per-frame chunked) HDF5 files and are
    read per batch so several jobs can share this PC with other experiments."""
    eps = []
    for path in paths:
        f = h5py.File(path, "r")
        for k in sorted(k for k in f.keys() if k.startswith("episode_")):
            if len(eps) >= n:
                break
            g = f[k]
            e = {"state": g["state"][()], "action": g["action"][()]}
            for c in CAMS:
                e[c] = g[c][()] if images_in_ram else g[c]
            eps.append(e)
    assert len(eps) == n, f"only {len(eps)} episodes available, wanted {n}"
    return eps


def compute_stats(eps: list[dict]) -> dict:
    S = np.concatenate([e["state"] for e in eps])
    A = np.concatenate([e["action"] for e in eps])
    stats = {
        "observation.state": {"mean": S.mean(0), "std": S.std(0) + 1e-3},
        "action": {"mean": A.mean(0), "std": A.std(0) + 1e-3},
    }
    for c in CAMS:
        px = np.concatenate([e[c][::15].reshape(-1, 3) for e in eps[:: max(1, len(eps) // 40)]]).astype(np.float32) / 255.0
        stats[f"observation.images.{c}"] = {"mean": px.mean(0).reshape(3, 1, 1), "std": px.std(0).reshape(3, 1, 1)}
    return {k: {s: torch.tensor(v, dtype=torch.float32) for s, v in d.items()} for k, d in stats.items()}


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
    """Random (episode, t) samples with ACT action chunks and padding flags.

    Images live on a slow HDD, so a buffer of `buffer` fully decoded episodes is kept in RAM and one
    episode is swapped in by a background thread every `refresh` steps (sequential reads only).
    """

    def __init__(self, eps: list[dict], chunk: int, rng: np.random.Generator, buffer: int = 16, refresh: int = 10):
        import queue
        import threading

        self.eps, self.chunk, self.rng = eps, chunk, rng
        self.lengths = np.array([len(e["action"]) for e in eps])
        self.refresh, self.calls = refresh, 0
        k = min(buffer, len(eps))
        self.slots = list(rng.choice(len(eps), size=k, replace=False))
        self.cache = [self._decode(i) for i in self.slots]
        self.q = queue.Queue(maxsize=2)
        self.stop = False

        def loader():
            lrng = np.random.default_rng(int(rng.integers(1 << 30)))
            while not self.stop:
                i = int(lrng.integers(len(self.eps)))
                self.q.put((i, self._decode(i)))

        if len(eps) > k:
            threading.Thread(target=loader, daemon=True).start()

    def _decode(self, i: int) -> dict:
        e = self.eps[i]
        return {c: np.asarray(e[c][()] if hasattr(e[c], "shape") and not isinstance(e[c], np.ndarray) else e[c])
                for c in CAMS}

    def _maybe_swap(self) -> None:
        self.calls += 1
        if self.calls % self.refresh or self.q.empty():
            return
        i, imgs = self.q.get_nowait()
        if i in self.slots:
            return
        j = int(self.rng.integers(len(self.slots)))
        self.slots[j], self.cache[j] = i, imgs

    def batch(self, bs: int, device: str) -> dict:
        self._maybe_swap()
        st, act, pad, imgs = [], [], [], {c: [] for c in CAMS}
        lens = np.array([self.lengths[i] for i in self.slots], dtype=float)
        for _ in range(bs):
            j = self.rng.choice(len(self.slots), p=lens / lens.sum())
            ei = self.slots[j]
            e = self.eps[ei]
            T = self.lengths[ei]
            t = self.rng.integers(0, T)
            idx = np.arange(t, t + self.chunk)
            is_pad = idx >= T
            idx = np.minimum(idx, T - 1)
            st.append(e["state"][t])
            act.append(e["action"][idx])
            pad.append(is_pad)
            for c in CAMS:
                imgs[c].append(self.cache[j][c][t])
        b = {
            "observation.state": torch.from_numpy(np.stack(st)),
            "action": torch.from_numpy(np.stack(act)),
            "action_is_pad": torch.from_numpy(np.stack(pad)),
        }
        for c in CAMS:
            x = torch.from_numpy(np.stack(imgs[c])).permute(0, 3, 1, 2).float() / 255.0
            b[f"observation.images.{c}"] = x
        return {k: v.to(device, non_blocking=True) for k, v in b.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--data", nargs="+", default=["data/stage{stage}.hdf5", "data/stage{stage}_more.hdf5"])
    ap.add_argument("--buffer", type=int, default=16, help="episodes kept decoded in RAM (~31 MB each)")
    ap.add_argument("--refresh", type=int, default=10, help="swap one buffered episode every N steps")
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
    eps = load_episodes([p for p in paths if p.exists()], args.episodes)
    img_hw = eps[0]["front"].shape[1:3]
    stats = compute_stats(eps)
    cfg = make_config(img_hw, device)
    policy = ACTPolicy(cfg, dataset_stats=stats).to(device)
    policy.train()
    opt = torch.optim.AdamW(policy.get_optim_params(), lr=cfg.optimizer_lr, weight_decay=cfg.optimizer_weight_decay)
    sampler = Sampler(eps, cfg.chunk_size, rng, buffer=args.buffer, refresh=args.refresh)
    n_params = sum(p.numel() for p in policy.parameters())
    meta = {"stage": args.stage, "episodes": len(eps), "frames": int(sampler.lengths.sum()), "steps": args.steps,
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
