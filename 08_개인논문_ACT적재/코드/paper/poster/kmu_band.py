"""KMU signature for the blue title band: shield in its own colours, white page background removed,
logotype '계명대학교' turned white (reverse use on a dark background). -> logos/kmu_signature_kor_band.png"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

D = Path(__file__).resolve().parent / "logos"
im = Image.open(D / "kmu_signature_kor.png").convert("RGB")
a = np.asarray(im).astype(float)
h, w, _ = a.shape
SPLIT = 258                                   # empty columns between the shield (12-227) and the text (290-)
out = np.zeros((h, w, 4), dtype=np.uint8)
# shield: keep colours, make the outside white transparent (white connected to the image border)
# .copy(): an image made by fromarray is read-only and floodfill on it silently does nothing
m = Image.fromarray(((a[:, :SPLIT].min(axis=2) > 225) * 255).astype(np.uint8)).copy()
for seed in [(0, 0), (SPLIT - 1, 0), (0, h - 1), (SPLIT - 1, h - 1)]:
    ImageDraw.floodfill(m, seed, 128)
bg = np.asarray(m) == 128
shield = a[:, :SPLIT].copy()
alpha = np.where(bg, 0.0, 1.0)
# anti-aliased rim next to the removed background is blue mixed with white: make it blue with partial alpha,
# otherwise a thin white line stays around the shield on the blue band
near = bg.copy()
for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
    near |= np.roll(np.roll(bg, dy, 0), dx, 1)
rim = near & ~bg
OUTLINE = np.array([0, 64, 144], float)          # the shield's outermost blue
t = np.clip((255 - shield[..., 0]) / 255, 0, 1)  # red channel: 255 for white, 0 for the outline blue
alpha = np.where(rim, np.minimum(alpha, t), alpha)
shield[rim] = OUTLINE
out[:, :SPLIT, :3] = shield.astype(np.uint8)
out[:, :SPLIT, 3] = (alpha * 255).astype(np.uint8)
# logotype: white, alpha from how dark the blue ink is (keeps anti-aliasing)
L = a[:, SPLIT:].mean(axis=2)
out[:, SPLIT:, :3] = 255
out[:, SPLIT:, 3] = (np.clip((255 - L) / (255 - 72), 0, 1) * 255).astype(np.uint8)
Image.fromarray(out, "RGBA").save(D / "kmu_signature_kor_band.png")
print("saved", (w, h))
