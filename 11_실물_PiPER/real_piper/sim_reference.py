#!/usr/bin/env python
"""실물 카메라를 시뮬과 같은 자리·방향에 달 때 볼 기준 사진 (CPU, 팔 필요 없음, 2026-11-04).

시뮬 기본 장면을 처음 자세로 띄워 위 카메라·손목 카메라 사진을 두 가지로 저장한다:
  *_view.png  — 보통 카메라가 보는 방향 (실물 카메라 화면과 이것을 나란히 놓고 맞춘다)
  *_model.png — VLA 에 들어가는 꼴 (위아래 뒤집힘, cameras.py 가 실물 사진을 이렇게 만든다)
위 카메라: 책상 윗면에서 0.85 m 위, 로봇 밑동 바로 위, 똑바로 아래를 보고, 사진 위쪽 = 로봇 앞(+x), 세로 시야각 75°.
  CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PIPER_NOPED=1 python real_piper/sim_reference.py --out real_piper/reference
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PIPER_NOPED", "1")
os.environ.setdefault("PIPER_WORLD", str(ROOT / "piper_sim/worlds/default.json"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "real_piper" / "reference"))
    p.add_argument("--seed", type=int, default=1000)
    a = p.parse_args()
    import piper_sim.piper_robot as pr
    pr.use_piper_in_lerobot()
    import piper_sim.world as W
    W.install()
    import piper_sim.world_expert as WE
    from PIL import Image
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sc = W.sample_scene(a.seed, n_objects=4, n_pads=2)
    ep = WE.make_episode(sc, a.seed)
    px = ep.obs["pixels"]
    for key, name in (("image", "top"), ("image2", "wrist")):
        img = np.asarray(px[key])
        Image.fromarray(img).save(out / f"{name}_model.png")
        Image.fromarray(np.ascontiguousarray(img[::-1])).save(out / f"{name}_view.png")
    cam = W.CFG["camera"]
    (out / "README.txt").write_text(
        f"위 카메라: 자리 {cam['pos']} (책상 {W.TABLE_Z} m), 보는 곳 {cam['lookat']}, 사진 위쪽 방향 {cam['up']}, 세로 시야각 {cam['fovy']}°\n"
        f"손목 카메라: 로봇 파일 eye_in_hand (손 끝판에 붙음, D405 크기), 세로 시야각 75°\n"
        f"처음 관절 (rad): {np.round(ep.obs['robot_state']['joints']['pos'], 4).tolist()}\n", encoding="utf-8")
    print("저장:", out)
    ep.env.close()


if __name__ == "__main__":
    main()
