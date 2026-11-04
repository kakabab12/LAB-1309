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
m = Image.fromarray(((a[:, :SPLIT].min(axis=2) > 225) * 255).astype(np.uint8))
for seed in [(0, 0), (SPLIT - 1, 0), (0, h - 1), (SPLIT - 1, h - 1)]:
    ImageDraw.floodfill(m, seed, 128)
bg = np.asarray(m) == 128
out[:, :SPLIT, :3] = a[:, :SPLIT].astype(np.uint8)
out[:, :SPLIT, 3] = np.where(bg, 0, 255)
# logotype: white, alpha from how dark the blue ink is (keeps anti-aliasing)
L = a[:, SPLIT:].mean(axis=2)
out[:, SPLIT:, :3] = 255
out[:, SPLIT:, 3] = (np.clip((255 - L) / (255 - 72), 0, 1) * 255).astype(np.uint8)
Image.fromarray(out, "RGBA").save(D / "kmu_signature_kor_band.png")
print("saved", (w, h))
