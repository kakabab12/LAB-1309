#!/usr/bin/env python
"""모은 시범(world_collect) 마다 VLA 물체 위치 입력(world_feat, 24칸)을 미리 계산해 *.wfeat.npy 로 (학습 train_lora --objfeat --feat wfeat).
   평가 때와 같은 길: 저장된 위 카메라 사진(JPEG) → world_detect → 지시문의 물체·놓을 곳 − 손.
   PIPER_NOPED=1 python piper_sim/world_feat_cache.py --data data/world_default_t [--shard 0 --nshard 3]"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import piper_sim.world as W  # noqa: E402
import piper_sim.world_feat as WF  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshard", type=int, default=1)
    a = p.parse_args()
    wf = WF.WorldFeat()
    files = sorted(f for d in a.data for f in Path(d, "episodes").glob("*.npz"))[a.shard::a.nshard]
    n = 0
    for f in files:
        out = f.with_suffix("").with_suffix(".wfeat.npy") if f.suffix == ".npz" else None
        out = f.parent / f"{f.stem}.wfeat.npy"
        if out.exists():
            continue
        ep = np.load(f, allow_pickle=True)
        task = str(ep["task"])
        meta = json.loads(str(ep["scene"])) if "scene" in ep.files else {}
        cam = meta.get("env_info", {}).get("camera")
        W.CAMERA_CALIB = (cam[0], cam[1]) if cam else None      # 흔든 장면이면 그 장면의 카메라 보정값
        wf.reset()
        feats = []
        J = ep["joints"] if "joints" in ep.files and len(ep["joints"]) == len(ep["img"]) else [None] * len(ep["img"])
        for blob, pos, q in zip(ep["img"], ep["eef_pos"], J):
            img = np.asarray(Image.open(io.BytesIO(bytes(blob))).convert("RGB"))
            feats.append(wf.step({"pixels": {"image": img}, "robot_state": {"eef": {"pos": pos}, "joints": {"pos": q}}}, task, hand=pos))
        np.save(out, np.stack(feats).astype(np.float32))
        n += 1
        if n % 50 == 0:
            print(f"{n}/{len(files)}", flush=True)
    print("끝", n, flush=True)


if __name__ == "__main__":
    main()
