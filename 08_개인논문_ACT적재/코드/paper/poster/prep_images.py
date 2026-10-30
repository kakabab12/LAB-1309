"""Poster image prep: paint the black void beyond the simulated table light grey, then crop each render to the
frame the poster uses. -> img/p_*.png"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

IMG = Path(__file__).resolve().parent / "img"
VOID = (236, 240, 245)


def unvoid(im):
    im = im.convert("RGB").copy()
    w, h = im.size
    a = np.asarray(im)
    for seed in [(x, 0) for x in range(0, w, 40)]:   # flood the pure-black sky from the top edge only
        if a[seed[1], seed[0]].max() <= 3:
            ImageDraw.floodfill(im, seed, VOID, thresh=6)
            a = np.asarray(im)
    return im


def crop(name, box, out):
    im = unvoid(Image.open(IMG / name)).crop(box)
    im.save(IMG / out)
    print(out, im.size)


def trim(name, pad=24):
    im = Image.open(IMG / name).convert("RGB")
    a = np.asarray(im).astype(int)
    ys, xs = np.where((a < 245).any(2))
    box = (max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, im.width), min(ys.max() + pad, im.height))
    im.crop(box).save(IMG / ("p_" + name))
    print("p_" + name, im.crop(box).size)


if __name__ == "__main__":
    for r in ("rule1.png", "rule2.png", "rule3.png"):
        trim(r, pad=6)
    crop("basic_after.png", (0, 230, 1600, 1130), "p_hero.png")              # 16:9, robot + scale + stack
    tile = lambda x0, y0: (x0, y0, x0 + 1400, y0 + 836)                      # 1.675:1 environment tiles
    crop("clutter.png", tile(120, 260), "p_clutter.png")
    crop("factory.png", tile(200, 250), "p_factory.png")
    crop("looks_3.png", tile(120, 260), "p_looks.png")
    crop("cup_after.png", tile(120, 260), "p_cup.png")
    crop("box_after.png", tile(120, 260), "p_box.png")
    if (IMG / "pallet.png").exists():
        crop("pallet.png", tile(180, 40), "p_pallet.png")
