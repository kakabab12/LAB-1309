#!/usr/bin/env python
"""모은 시범(world_collect) 마다 VLA 물체 위치 입력(world_feat, 24칸)을 미리 계산해 *.wfeat.npy 로 (학습 train_lora --objfeat --feat wfeat).
   평가 때와 같은 길: 저장된 위 카메라 사진(JPEG) → world_detect → 지시문의 물체·놓을 곳 − 손.
   PIPER_NOPED=1 python piper_sim/world_feat_cache.py --data data/world_default_t [--shard 0 --nshard 3]"""
import argparse
import io
import zlib
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
    p.add_argument("--version", type=int, default=1, help="world_feat 판 (2: 옮길 물체 방향 [8:11], 3: + 가까이 보기 [11:15])")
    p.add_argument("--sure", action="store_true", help="10/27: 덜 확실한 찾기(fallback)는 기억을 덮지 않음 → 파일 이름 끝에 s (*.wfeat2s.npy)")
    p.add_argument("--from-v2", action="store_true", help="3판을 사진 없이 *.wfeat2.npy 에서 바로 (world_feat.derive_v3, 평가와 같은 식)")
    p.add_argument("--reverse", action="store_true", help="뒤에서부터 (앞에서부터 도는 줄과 겹치지 않게 늦게 더 붙일 때)")
    p.add_argument("--stride", type=int, default=1, help="학습이 읽는 장만 계산 (train_lora --frame-stride 와 같게; 사이 장은 앞 값)")
    a = p.parse_args()
    wf = WF.WorldFeat(version=a.version, sure=a.sure)
    suf = "wfeat" if a.version == 1 else f"wfeat{a.version}"      # 2판은 *.wfeat2.npy (학습 중인 1판 파일을 건드리지 않게)
    suf += "s" if a.sure else ""
    files = sorted(f for d in a.data for f in Path(d, "episodes").glob("*.npz"))[a.shard::a.nshard]
    if a.reverse:
        files = files[::-1]
    n = 0
    for f in files:
        out = f.with_suffix("").with_suffix(".wfeat.npy") if f.suffix == ".npz" else None
        out = f.parent / f"{f.stem}.{suf}.npy"
        if out.exists():
            continue
        if a.from_v2:
            np.save(out, WF.derive_v3(np.load(f.parent / f"{f.stem}.wfeat2{'s' if a.sure else ''}.npy")))
            n += 1
            continue
        ep = np.load(f, allow_pickle=True)
        task = str(ep["task"])
        meta = json.loads(str(ep["scene"])) if "scene" in ep.files else {}
        cam = meta.get("env_info", {}).get("camera")
        W.CAMERA_CALIB = (cam[0], cam[1]) if cam else None      # 흔든 장면이면 그 장면의 카메라 보정값
        wf.reset()
        feats = []
        J = ep["joints"] if "joints" in ep.files and len(ep["joints"]) == len(ep["img"]) else [None] * len(ep["img"])
        off = zlib.crc32(f.stem.encode()) % a.stride if a.stride > 1 else 0     # train_lora.vis_offset 와 같은 식
        last = None
        for t, (blob, pos, q) in enumerate(zip(ep["img"], ep["eef_pos"], J)):
            if t % a.stride == off or last is None and t >= off:
                img = np.asarray(Image.open(io.BytesIO(bytes(blob))).convert("RGB"))
                last = wf.step({"pixels": {"image": img}, "robot_state": {"eef": {"pos": pos}, "joints": {"pos": q}}}, task, hand=pos)
            feats.append(last if last is not None else np.zeros(24, np.float32))
        np.save(out, np.stack(feats).astype(np.float32))
        n += 1
        if n % 50 == 0:
            print(f"{n}/{len(files)}", flush=True)
    print("끝", n, flush=True)


if __name__ == "__main__":
    main()
