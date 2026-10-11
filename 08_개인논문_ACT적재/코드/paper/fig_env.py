"""Fig. 2 (new): simulation cell (a-c, from fig2_sim.png) + generalisation objects and the factory cell (d-f).
usage: python paper/fig_env.py  -> paper/fig2_env.png"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
top = Image.open(ROOT / "paper/fig2_sim.png").convert("RGB")
W = top.width
gap = 8


def after(tile):  # right half (after the 3 stages) of an object_views tile, without the label bar
    im = Image.open(ROOT / "results/obj_tiles" / tile).convert("RGB")
    return im.crop((400 + 40, 40, 800 - 20, 300))


fv = Image.open(ROOT / "results/factory_views.png").convert("RGB")
fac = fv.crop((600, 316, 1000, 600))
panels = [after("cup_orig.png"), after("box_orig.png"), fac]
pw = (W - 2 * gap) // 3
ph = int(pw * 0.78)
row = Image.new("RGB", (W, ph), "white")
x = 0
# panel names in the picture itself (10-11: the bare (a)-(f) letters were too small to read at column width)
FONT = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 30)
SMALL = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 27)


def tag(d: ImageDraw.ImageDraw, xy, text: str, font=FONT) -> None:
    x0, y0 = xy
    l, t, r, b = d.textbbox((x0 + 9, y0 + 3), text, font=font)
    d.rounded_rectangle((x0, y0, r + 9, b + 7), radius=6, fill="white", outline=(60, 60, 60), width=2)
    d.text((x0 + 9, y0 + 3), text, fill=(20, 20, 20), font=font)


def callout(d: ImageDraw.ImageDraw, xy, to, text: str) -> None:
    d.line((xy[0] + 40, xy[1] + 4, *to), fill="white", width=6)
    d.line((xy[0] + 40, xy[1] + 4, *to), fill=(20, 20, 20), width=3)
    d.ellipse((to[0] - 6, to[1] - 6, to[0] + 6, to[1] + 6), fill=(20, 20, 20), outline="white", width=2)
    tag(d, xy, text, SMALL)


for k, p in enumerate(panels):
    p = p.resize((pw, ph))
    tag(ImageDraw.Draw(p), (4, 4), f"({'def'[k]}) {['손잡이 컵', '직사각형 상자', '공장형 환경'][k]}")
    row.paste(p, (x, 0))
    x += pw + gap
out = Image.new("RGB", (W, top.height + gap + ph), "white")
out.paste(top, (0, 0))
out.paste(row, (0, top.height + gap))
d = ImageDraw.Draw(out)
tag(d, (4, 4), "(a) 시뮬레이션 환경")  # covers the old letter boxes of fig2_sim.png
tag(d, (652, 4), "(b) 손목 카메라")
tag(d, (652, 244), "(c) 상단 카메라")
callout(d, (20, 400), (185, 280), "저울 (집는 곳)")
callout(d, (330, 415), (440, 360), "팔레트 (쌓는 곳)")
out.save(ROOT / "paper/fig2_env.png")
print(out.size)
