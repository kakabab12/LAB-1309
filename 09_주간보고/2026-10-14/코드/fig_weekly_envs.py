"""Montage for the weekly report: the capstone stacking system rebuilt in simulation and run in various environments.
Numbers are simulation success rates from the paper experiments (08_개인논문_ACT적재 journals 10/7-10/11)."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "LAB-1309" / "08_개인논문_ACT적재" / "결과" / "그림"
OUT = HERE / "weekly_1014_envs.png"
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 25)
FONT_N = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 27)

ov = Image.open(FIG / "object_views.png").convert("RGB")
tile = lambda r, c: ov.crop((c * 400, r * 300 + 23, (c + 1) * 400, (r + 1) * 300))  # object_views: 3 rows x 4, label strip on top
PANELS = [
    ("Basic", "99.0%", tile(0, 1)),
    ("Cluttered table", "92%", Image.open(FIG / "clutter_views.png").convert("RGB").crop((0, 325, 400, 600))),
    ("Factory-like scene", "90%", Image.open(FIG / "factory_views.png").convert("RGB").crop((0, 17, 400, 300))),
    ("Cup with a handle", "88%", tile(1, 1)),
    ("Rectangular box", "100%", tile(2, 1)),
    ("8-bin pallet (2x2x2)", "95%", Image.open(FIG / "pallet_2x2x2_final_s0_final.png").convert("RGB").crop((40, 20, 440, 315))),
]
TW, TH, LH, GAP = 400, 280, 46, 14
W = 3 * TW + 2 * GAP
H = 2 * (TH + LH) + GAP
out = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(out)
for k, (name, num, im) in enumerate(PANELS):
    r, c = divmod(k, 3)
    x, y = c * (TW + GAP), r * (TH + LH + GAP)
    s = max(TW / im.width, TH / im.height)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    left, top = (im.width - TW) // 2, (im.height - TH) // 2
    out.paste(im.crop((left, top, left + TW, top + TH)), (x, y))
    d.rectangle((x, y + TH, x + TW - 1, y + TH + LH - 1), fill=(235, 241, 250))
    d.text((x + 10, y + TH + 10), name, font=FONT, fill=(20, 20, 20))
    nw = d.textlength(num, font=FONT_N)
    d.text((x + TW - nw - 10, y + TH + 8), num, font=FONT_N, fill=(11, 61, 145))
out.save(OUT)
print(OUT, out.size)
