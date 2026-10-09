"""Shrink a GIF for the repo: every 2nd frame, given width, 96 colours.  usage: python shrink_gif.py src dst [width]"""
import sys

from PIL import Image, ImageSequence

src, dst = sys.argv[1], sys.argv[2]
width = int(sys.argv[3]) if len(sys.argv) > 3 else 480
im = Image.open(src)
fr = []
for i, f in enumerate(ImageSequence.Iterator(im)):
    if i % 2 == 0:
        f = f.convert("RGB")
        fr.append(f.resize((width, round(f.height * width / f.width))).quantize(colors=96))
fr[0].save(dst, save_all=True, append_images=fr[1:], duration=200, loop=0, optimize=True)
