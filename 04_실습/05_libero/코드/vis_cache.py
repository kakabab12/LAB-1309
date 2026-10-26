#!/usr/bin/env python
"""
사진 특징 미리 계산해 두기 — 학습 속도 올리기 (2026-10-02)

왜
  1080 Ti 에서 학습은 초당 약 4.6장밖에 못 본다. 한 장마다 사진 2장을 SigLIP 비전 인코더(512×512, 1024패치)에
  통과시키는 계산이 가장 크다. 그런데 비전 인코더는 학습하지 않는다(얼려 둠) — 같은 사진이면 결과가 늘 같다.
  그래서 한 번만 계산해 디스크에 두고, 학습 때는 읽어서 쓴다.

  잃는 것: 사진 증강(밝기·위치 흔들기). 시뮬레이터 정확도에는 영향이 작다. 실물 로봇 단계에서는 증강을 다시 켠다.

쓰는 법
  python vis_cache.py build --policy outputs/v6b_model/merged --data data/v6_normal data/v6_switch --out cache/vis
  학습: python train_lora.py ... --vis-cache cache/vis
"""
import argparse
import io
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

STATE = {"feats": None}
SHAPE = (2, 64, 960)


def install(policy):
    """embed_image 를 바꿔, STATE['feats'] 가 있으면 계산 대신 그 값을 차례로 돌려준다."""
    vwe = policy.model.vlm_with_expert
    if getattr(vwe, "_vis_cache_installed", False):
        return
    orig = vwe.embed_image

    def embed_image(image):
        f = STATE["feats"]
        if f:
            return f.pop(0).to(device=image.device, dtype=torch.float32)
        return orig(image)
    vwe.embed_image = embed_image
    vwe._vis_cache_installed = True


def cache_path(root, npz):
    npz = Path(npz)
    return Path(root) / npz.parent.parent.name / f"{npz.stem}.npy"


def decode(blob):
    b = blob if isinstance(blob, (bytes, bytearray)) else bytes(blob)
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))


@torch.inference_mode()
def build(a):
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.env_processor import LiberoProcessorStep
    dev = "cuda"
    pol = SmolVLAPolicy.from_pretrained(a.policy)
    pol.config.device = dev
    pol.to(dev).eval()
    pre, _ = make_pre_post_processors(policy_cfg=pol.config, pretrained_path=a.policy,
                                      preprocessor_overrides={"device_processor": {"device": dev}})
    env_step = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
    objfeat = (Path(a.policy) / "objfeat").exists()
    files = sorted(f for d in a.data for f in Path(d).glob("episodes/*.npz"))
    files = files[a.shard::a.nshard]
    if getattr(a, "reverse", False):     # 10/25: 앞에서부터 도는 줄에 뒤에서부터 하나 더 붙일 때 (가운데서 만남)
        files = files[::-1]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    sf = Path(a.out) / "stride.txt"
    if sf.exists():
        # 10/25: 저장 간격의 배수면 허용 (crc % 6 ≡ crc % 3 (mod 3) 이라 6장 간격 프레임은 3장 간격 프레임 안에 있다 — 학습 --frame-stride 6)
        assert a.stride % int(sf.read_text()) == 0, f"{a.out} 는 {sf.read_text()}장마다 계산본 — --stride 는 그 배수로"
    else:
        sf.write_text(str(a.stride))
    t0, nfr, done = time.time(), 0, 0
    for f in files:
        dst = cache_path(a.out, f)
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        d = np.load(f, allow_pickle=True)
        n = len(d["action"])
        tmp = dst.with_suffix(f".tmp{os.getpid()}.npy")      # 두 줄이 같은 파일을 동시에 써도 서로 덮지 않게 (끝에 이름 바꾸기)
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n,) + SHAPE)
        import zlib
        off = zlib.crc32(Path(f).stem.encode()) % a.stride if a.stride > 1 else 0   # train_lora.vis_offset 와 같다
        frames = list(range(off, n, a.stride))
        for i in range(0, len(frames), a.batch):
            ts = frames[i:i + a.batch]
            obs = {"pixels": {"image": np.stack([decode(d["img"][t]) for t in ts]),
                              "image2": np.stack([decode(d["wrist"][t]) for t in ts])},
                   "robot_state": {"eef": {"pos": d["eef_pos"][ts], "quat": d["eef_quat"][ts]},
                                   "gripper": {"qpos": d["grip"][ts]}}}
            b = preprocess_observation(obs)
            b["task"] = [str(d["task"])] * len(ts)
            b = env_step(b)
            if objfeat:     # 10/14: 위치 입력 모델은 상태 32칸 — 사진 특징과 무관하므로 0 으로 채워 정규화만 통과시킨다
                st = b["observation.state"]
                b["observation.state"] = torch.cat([st, torch.zeros(st.shape[0], 24, dtype=st.dtype, device=st.device)], -1)
            b = pre(b)
            imgs, _ = pol.prepare_images(b)
            e = torch.stack([pol.model.vlm_with_expert.embed_image(im) for im in imgs], 1)
            mm[ts] = e.half().cpu().numpy()
        mm.flush()
        del mm
        tmp.rename(dst)
        done += 1
        nfr += n
        if done % 25 == 0:
            el = time.time() - t0
            print(f"{done}/{len(files)} 에피소드, {nfr / el:.1f} 프레임/초", flush=True)
    print(f"끝: {done} 에피소드, {nfr} 프레임, {time.time() - t0:.0f}초", flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--policy", required=True)
    b.add_argument("--data", nargs="+", required=True)
    b.add_argument("--out", default="cache/vis")
    b.add_argument("--batch", type=int, default=16)
    b.add_argument("--shard", type=int, default=0)
    b.add_argument("--nshard", type=int, default=1)
    b.add_argument("--stride", type=int, default=1, help="이 장마다 하나만 계산 (학습 --frame-stride 와 같게, 10/14)")
    b.add_argument("--reverse", action="store_true", help="뒤에서부터")
    a = p.parse_args()
    build(a)


if __name__ == "__main__":
    main()
