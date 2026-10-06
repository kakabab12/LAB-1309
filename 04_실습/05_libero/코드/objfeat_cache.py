#!/usr/bin/env python
"""시범 사진마다 색·모양 검출(objdet.detect) 결과를 미리 계산해 에피소드 옆에 둔다 (2026-10-06).
   data/X/episodes/N8_ep2002.npz → data/X/episodes/N8_ep2002.objfeat.npy  (프레임 수, 24)
   python objfeat_cache.py --data data/piper_blk_normal_t ... --shard 0 --nshard 6"""
import argparse
import io
import time
from pathlib import Path

import numpy as np
from PIL import Image

import objdet


def decode(b):
    return np.asarray(Image.open(io.BytesIO(bytes(b))).convert("RGB"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshard", type=int, default=1)
    a = p.parse_args()
    files = sorted(f for d in a.data for f in Path(d).glob("episodes/*.npz"))[a.shard::a.nshard]
    t0, done, nfr = time.time(), 0, 0
    for f in files:
        dst = f.with_suffix(".objfeat.npy")
        if dst.exists():
            continue
        d = np.load(f, allow_pickle=True)
        n = len(d["action"])
        out = np.stack([objdet.detect(decode(d["img"][t]), decode(d["wrist"][t])) for t in range(n)]).astype(np.float32)
        tmp = dst.with_suffix(".tmp.npy")
        np.save(tmp, out)
        tmp.rename(dst)
        done += 1
        nfr += n
        if done % 200 == 0:
            print(f"{done}/{len(files)} 에피소드, {nfr / (time.time() - t0):.0f} 프레임/초", flush=True)
    print(f"끝: {done} 에피소드, {nfr} 프레임, {time.time() - t0:.0f}초", flush=True)


if __name__ == "__main__":
    main()
