#!/usr/bin/env python
"""여러 모양·N개 장면 보기 (10/24): 장면을 뽑아 띄우고 위 카메라·손목 카메라·비스듬한 시점 사진을 한 장으로.
   PIPER_NOPED=1 MUJOCO_GL=egl python piper_sim/world_view.py --seeds 1 2 3 --n 4 8 12 --out outputs/media/world_view.png"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import piper_sim.piper_robot as pr  # noqa: E402

pr.use_piper_in_lerobot()
import piper_sim.world as W  # noqa: E402

W.install()


def shots(env, big=480):
    inner = env._env.env
    sim = inner.sim
    out = []
    for cam in (W.CFG["camera"]["name"], "robot0_eye_in_hand", "galleryview"):
        try:
            im = sim.render(camera_name=cam, width=big, height=big)[::-1]
        except Exception as e:      # 카메라 이름이 다르면 건너뜀
            print("카메라", cam, e)
            im = np.zeros((big, big, 3), np.uint8)
        out.append(im)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--n", type=int, nargs="+", default=[4, 8, 12])
    p.add_argument("--pads", type=int, default=None)
    p.add_argument("--out", default="outputs/media/world_view.png")
    a = p.parse_args()
    reach = W.reach_map()
    ok = [i for i, r in enumerate(reach["r_min"]) if r is not None]
    print(f"1번 관절 {reach['joint1_deg'][0]:.0f}~{reach['joint1_deg'][1]:.0f}°, 닿는 각도 {reach['theta'][ok[0]]:.0f}~{reach['theta'][ok[-1]]:.0f}° "
          f"({len(ok)}/{len(reach['theta'])} 칸), 반지름 {min(reach['r_min'][i] for i in ok):.2f}~{max(reach['r_max'][i] for i in ok):.2f}m", flush=True)
    rows = []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
    plt.rcParams["font.family"] = "NanumGothic"
    for seed, n in zip(a.seeds, a.n):
        sc = W.sample_scene(seed, n_objects=n, n_pads=a.pads)
        print(f"장면 {seed}: 물체 {len(sc.objects)} {[W.describe(o) for o in sc.objects]}, 판 {len(sc.pads)}", flush=True)
        env = W.make_env(sc)
        env.reset(seed=seed)
        W.place(env, sc)
        rows.append((sc, shots(env)))
        tk = W.tasks_for(sc)
        print(f"  할 수 있는 일 {len(tk)}개, 예: {tk[0][2]} / {tk[-1][2]}", flush=True)
        env.close()
    fig, ax = plt.subplots(len(rows), 3, figsize=(12, 4 * len(rows)))
    ax = np.atleast_2d(ax)
    for i, (sc, ims) in enumerate(rows):
        for j, (im, t) in enumerate(zip(ims, ["위 카메라", "손목 카메라", "비스듬히"])):
            ax[i, j].imshow(im)
            ax[i, j].set_title(f"장면 {sc.seed} · 물체 {len(sc.objects)} 판 {len(sc.pads)} · {t}" if j == 0 else t, fontsize=10)
            ax[i, j].axis("off")
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=100)
    print("그림:", a.out)


if __name__ == "__main__":
    main()
