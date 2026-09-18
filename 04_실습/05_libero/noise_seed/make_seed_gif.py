#!/usr/bin/env python
"""같은 상태·같은 지시, 노이즈만 다를 때의 차이를 나란히 보여주는 GIF.

오늘(2026-09-18)의 핵심 발견을 눈으로 보여주기 위한 것. 같은 초기상태(에피소드 10)에서
좋은 시드는 일을 끝내고, 나쁜 시드는 좁은 영역에서 맴돈다.
"""
import os
import sys

import numpy as np
import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont

F = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf", 15)
F2 = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf", 12)
PH = {"A": ("A 수행", "#2a78d6"), "R": ("스크립트", "#8a8984"),
      "B": ("B 수행 (지시 바뀐 뒤)", "#eb6834"), "A2": ("A 재개", "#1baf7a")}


def load(video, traj, stride):
    v = iio.imread(video, plugin="pyav")[:, :, :256]
    ph = np.load(traj)["phase"]
    return v[::stride], ph[::stride]


def side_by_side(items, out, stride=6, size=230, fps=10, colors=64):
    loaded = [(*load(v, t, stride), cap) for v, t, cap in items]
    n = max(len(v) for v, _, _ in loaded)
    W, H = size * len(loaded) + 10 * (len(loaded) - 1), size + 46
    frames = []
    for i in range(n):
        c = Image.new("RGB", (W, H), "#fcfcfb")
        d = ImageDraw.Draw(c)
        for k, (v, ph, cap) in enumerate(loaded):
            j = min(i, len(v) - 1)
            x = k * (size + 10)
            c.paste(Image.fromarray(v[j]).resize((size, size), Image.LANCZOS), (x, 28))
            lab, col = PH[str(ph[j])]
            d.text((x + 2, 1), cap, font=F, fill="#0b0b0b")
            d.rectangle([x, 24, x + size, 27], fill=col)
            d.text((x + 2, size + 30), f"{lab}   {j * stride}스텝", font=F2, fill="#52514e")
        frames.append(c.quantize(colors=colors, method=Image.MEDIANCUT))
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
    print(out, len(frames), f"{os.path.getsize(out) / 1e6:.1f}MB")


if __name__ == "__main__":
    ep = sys.argv[1] if len(sys.argv) > 1 else "10"
    m = "outputs/media_seed"
    items = []
    for tag, cap in [(f"A8_B7_grasp3_flush_ns10679_ep{ep}", "좋은 노이즈"),
                     (f"A8_B7_grasp3_flush_ep{ep}", "무작위 (기본)"),
                     (f"A8_B7_grasp3_flush_ns10485_ep{ep}", "나쁜 노이즈")]:
        v, t = f"{m}/videos/{tag}.mp4", f"{m}/traj/{tag}.npz"
        if os.path.exists(v) and os.path.exists(t):
            items.append((v, t, cap))
        else:
            print("없음:", v)
    if len(items) >= 2:
        side_by_side(items, "outputs/report/noise_seed_compare.gif")
    else:
        print("영상이 아직 없습니다. 큐 9단계가 끝난 뒤 실행하세요.")
