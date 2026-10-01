"""정책이 주어진 시범 데이터의 동작을 얼마나 잘 따라 하는지 (flow matching 손실). 학습 없이 평가만.
같은 데이터·같은 노이즈 시드로 원래 모델과 학습 모델을 비교한다."""
import argparse
import glob
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_lora as tl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--files", nargs="+", required=True, help="glob 패턴")
    p.add_argument("--max", type=int, default=40)
    p.add_argument("--batches", type=int, default=30)
    a = p.parse_args()
    files = sorted(f for g in a.files for f in glob.glob(g))
    random.Random(0).shuffle(files)
    files = files[:a.max]
    import sys; sys.argv = sys.argv[:1]; args = tl.parse_args()
    args.policy, args.lora_r = a.policy, 4
    t = tl.Trainer(args)
    ds = tl.ChunkDataset(files, t.chunk_size)
    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=tl.collate, num_workers=1,
                    generator=torch.Generator().manual_seed(0))
    torch.manual_seed(0)
    t.policy.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i, b in enumerate(dl):
            if i >= a.batches:
                break
            tot += t.loss(b).item(); n += 1
    print(f"LOSS {a.policy} {a.files} {tot / n:.4f} ({len(files)} 에피소드, {n} 배치)", flush=True)


if __name__ == "__main__":
    main()
