"""Fig. 5 (paper): the learned pallet policies stacking 2x2x2 (8 slots in a row) in simulation.

Frames come from the evaluation GIF of a fully successful sequence of the last pallet round (pallet6 if evaluated,
else pallet5): (a) right after the first layer (the 5th bin has just appeared on the scale), (b) the last frame.
A new bin appears on the scale at the start of every slot, so slot starts are found as rising edges of the blue
pixel count over the scale. -> paper/fig5_pallet.png
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageSequence

ROOT = Path(__file__).resolve().parent.parent
FONT = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 26)
SCALE_BOX = (140, 165, 240, 255)  # the scale in the 400x300 overview part of the GIF frame


def pick_gif() -> Path:
    for tag in ("pallet6_k25_p0", "pallet5_k25_p0"):
        g = sorted((ROOT / "results" / "eval" / tag / "gifs").glob("pallet_t*_oooooooo.gif"))
        if g and (ROOT / "results" / "eval" / tag / "results.json").exists():
            return g[0]
    raise SystemExit("no fully successful pallet GIF")


def blue_on_scale(f: Image.Image) -> int:
    a = np.asarray(f.crop(SCALE_BOX)).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return int(((b > 90) & (b > r + 40) & (b > g + 25)).sum())


def tag(im: Image.Image, text: str) -> None:
    d = ImageDraw.Draw(im)
    l, t, r, b = d.textbbox((12, 6), text, font=FONT)
    d.rounded_rectangle((4, 4, r + 8, b + 6), radius=5, fill="white", outline=(60, 60, 60), width=2)
    d.text((12, 6), text, font=FONT, fill=(20, 20, 20))


def main() -> None:
    gif = pick_gif()
    frames = [f.convert("RGB").crop((0, 0, 400, 300)) for f in ImageSequence.Iterator(Image.open(gif))]
    blue = np.array([blue_on_scale(f) for f in frames])
    on = blue > 0.5 * np.percentile(blue, 95)
    starts = [0]
    for i in range(1, len(on)):  # rising edges, at least 40 frames apart (the arm carrying a bin over the scale)
        if on[i] and not on[i - 1] and i - starts[-1] >= 40:
            starts.append(i)
    if len(starts) < 5:
        raise SystemExit(f"found {len(starts)} slot starts in {gif.name}")
    a = frames[min(starts[4] + 3, len(frames) - 1)].copy()
    b = frames[-1].copy()
    tag(a, "(a) 1층 4칸 완료")
    tag(b, "(b) 2층까지 8칸 완료")
    gap = 8
    out = Image.new("RGB", (800 + gap, 300), "white")
    out.paste(a.crop((0, 0, 400, 300)), (0, 0))
    out.paste(b.crop((0, 0, 400, 300)), (400 + gap, 0))
    out.save(ROOT / "paper" / "fig5_pallet.png")
    (ROOT / "paper" / "fig5_pallet.txt").write_text(f"{gif.relative_to(ROOT)} slot starts {starts}\n")
    print(gif, starts)


if __name__ == "__main__":
    main()
