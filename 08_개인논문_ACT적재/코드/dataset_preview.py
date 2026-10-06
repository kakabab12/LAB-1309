"""Previews of the demonstration datasets: one GIF and one contact sheet per file (what ACT sees).

usage: python dataset_preview.py data/stage1.hdf5 [data/stage2.hdf5 ...] --out results/preview
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 14)


def frame(f, t):
    a = np.concatenate([f["front"][t], f["top"][t]], 1)  # 120 x 320
    return Image.fromarray(a).resize((640, 240), Image.NEAREST)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out", default="results/preview")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in args.files:
        p = Path(p)
        with h5py.File(p, "r") as h:
            g = h[f"episode_{args.episode:04d}"]
            T = len(g["action"])
            frames = [frame(g, t) for t in range(0, T, 3)]
            frames[0].save(out / f"{p.stem}_ep{args.episode}.gif", save_all=True, append_images=frames[1:],
                           duration=100, loop=0, optimize=True)
            picks = np.linspace(0, T - 1, 6).astype(int)
            sheet = Image.new("RGB", (640 * 2, 240 * 3), "white")
            for i, t in enumerate(picks):
                im = frame(g, t)
                d = ImageDraw.Draw(im)
                d.rectangle([4, 4, 150, 24], fill=(255, 255, 255))
                d.text((8, 5), f"t={t / 30:.1f}s  front | top", font=FONT, fill=(0, 0, 0))
                sheet.paste(im, ((i % 2) * 640, (i // 2) * 240))
            sheet.save(out / f"{p.stem}_ep{args.episode}_frames.png")
            n = sum(1 for k in h if k.startswith("episode_"))
            print(p.name, "episodes", n, "len", T, "->", out)


if __name__ == "__main__":
    main()
