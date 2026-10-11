"""Fig. 4 at column width: the rollout image (paper/fig4_rollout.png, fig_rollout.py) with larger panel names.
Panels are 480x360 with 6 px gaps; the new label boxes cover the old small ones. -> paper/fig4_rollout_big.png"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONT = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 40)
LABELS = ["(a) 대기", "(b) 1층 집기", "(c) 1층 적재", "(d) 옆 적재", "(e) 2층 집기", "(f) 2층 적재"]

im = Image.open(ROOT / "paper" / "fig4_rollout.png").convert("RGB")
d = ImageDraw.Draw(im)
for k, text in enumerate(LABELS):
    x0, y0 = (k % 3) * 486 + 4, (k // 3) * 366 + 4
    l, t, r, b = d.textbbox((x0 + 10, y0 + 2), text, font=FONT)
    d.rounded_rectangle((x0, y0, r + 10, b + 8), radius=6, fill="white", outline=(60, 60, 60), width=2)
    d.text((x0 + 10, y0 + 2), text, font=FONT, fill=(20, 20, 20))
im.save(ROOT / "paper" / "fig4_rollout_big.png")
print(im.size)
