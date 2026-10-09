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
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
except OSError:
    font = ImageFont.load_default()
for k, p in enumerate(panels):
    p = p.resize((pw, ph))
    d = ImageDraw.Draw(p)
    d.rectangle((6, 6, 58, 40), fill="white")
    d.text((12, 8), f"({'def'[k]})", fill="black", font=font)
    row.paste(p, (x, 0))
    x += pw + gap
out = Image.new("RGB", (W, top.height + gap + ph), "white")
out.paste(top, (0, 0))
out.paste(row, (0, top.height + gap))
out.save(ROOT / "paper/fig2_env.png")
print(out.size)
