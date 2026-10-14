"""Compact simulation figure (panels a-c of Fig. 2) for the short versions: Korean and English labels.
-> paper/prof/fig_sim_top.png, paper/prof/fig_sim_top_en.png"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
FONT = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 30)
SMALL = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 27)
LABELS = {"": ["(a) 시뮬레이션 환경", "(b) 손목 카메라", "(c) 상단 카메라", "저울 (집는 곳)", "팔레트 (쌓는 곳)"],
          "_en": ["(a) Simulated system", "(b) Wrist camera", "(c) Top camera", "Scale (pick)", "Pallet (place)"]}


def tag(d, xy, text, font=FONT):
    x0, y0 = xy
    l, t, r, b = d.textbbox((x0 + 9, y0 + 3), text, font=font)
    d.rounded_rectangle((x0, y0, r + 9, b + 7), radius=6, fill="white", outline=(60, 60, 60), width=2)
    d.text((x0 + 9, y0 + 3), text, fill=(20, 20, 20), font=font)


def callout(d, xy, to, text):
    d.line((xy[0] + 40, xy[1] + 4, *to), fill="white", width=6)
    d.line((xy[0] + 40, xy[1] + 4, *to), fill=(20, 20, 20), width=3)
    d.ellipse((to[0] - 6, to[1] - 6, to[0] + 6, to[1] + 6), fill=(20, 20, 20), outline="white", width=2)
    tag(d, xy, text, SMALL)


for suffix, (a, b, c, scale, pallet) in LABELS.items():
    im = Image.open(ROOT / "paper" / "fig2_sim.png").convert("RGB")  # 968 x 480, small (a) (b) (c) letters
    d = ImageDraw.Draw(im)
    tag(d, (4, 4), a); tag(d, (652, 4), b); tag(d, (652, 244), c)
    callout(d, (20, 400), (185, 280), scale)
    callout(d, (330, 415), (440, 360), pallet)
    im.save(ROOT / "paper" / "prof" / f"fig_sim_top{suffix}.png")
    print(im.size, suffix or "ko")
