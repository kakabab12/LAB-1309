"""Prepare the two logos for the poster.
IEIE: white-on-navy CI from the cover of the official brochure (theieie.org/download/brochure_kor_20260109.pdf),
      turned into a white logo with transparency, upscaled 5x with smooth edges.
KMU:  official signature (교표 방패형, 좌우조합, 한글) from kmu.ac.kr 로고마크 자료실, white margins trimmed."""
import sys
import numpy as np
from PIL import Image

cover, kmu_jpg, outdir, prev = sys.argv[1:5]
im = Image.open(cover).convert("RGB")
c = np.asarray(im.crop((520, 1800, 960, 1920))).astype(float)
L = c.mean(axis=2)
ys, xs = np.where(L > 128)
x0, y0, x1, y1 = xs.min() - 8, ys.min() - 8, xs.max() + 9, ys.max() + 9
L = L[y0:y1, x0:x1]
Lb = np.median(L[L < 60]); Lw = np.percentile(L[L > 200], 60)
print("IEIE crop", L.shape, "bg", Lb, "white", Lw)
a = np.clip((L - Lb) / (Lw - Lb), 0, 1)
A = Image.fromarray((a * 255).astype("uint8"))
k = 5
a2 = np.asarray(A.resize((A.width * k, A.height * k), Image.LANCZOS)).astype(float) / 255
lo, hi = 0.10, 0.50
t = np.clip((a2 - lo) / (hi - lo), 0, 1); t = t * t * (3 - 2 * t)
out = np.zeros(t.shape + (4,), dtype="uint8"); out[..., :3] = 255; out[..., 3] = (t * 255).round().astype("uint8")
o = Image.fromarray(out, "RGBA"); o.save(f"{outdir}/ieie_logo_white.png"); print("IEIE out", o.size)
bg = Image.new("RGBA", o.size, (0x37, 0x60, 0x92, 255)); bg.alpha_composite(o)
bg.convert("RGB").save(f"{prev}/ieie_on_band.png")
bg.convert("RGB").crop((0, 0, 900, o.height)).save(f"{prev}/ieie_on_band_left.png")

kimg = Image.open(kmu_jpg).convert("RGB")
kk = np.asarray(kimg).astype(int)
ys, xs = np.where(kk.min(axis=2) < 235)
pad = 12
kc = kimg.crop((max(0, xs.min() - pad), max(0, ys.min() - pad), min(kimg.width, xs.max() + pad + 1), min(kimg.height, ys.max() + pad + 1)))
kc.save(f"{outdir}/kmu_signature_kor.png"); print("KMU out", kc.size, round(kc.width / kc.height, 2))
