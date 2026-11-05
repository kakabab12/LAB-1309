"""Weekly report 11-12 picture: the 3D-printed U20CAM wrist camera mount on the PiPER gripper (simulation render).
Left: the whole arm, right: close-up. Renders: LAB-1309 11_실물_PiPER/손목카메라_마운트_U20CAM/img/70_{overview,closeup}.png (cad/sim_check.py)."""
import sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

IMG, OUT = Path(sys.argv[1]), sys.argv[2]
over = np.asarray(Image.open(IMG / "70_overview.png").convert("RGB"))[270:1150, 520:1300]
close = np.asarray(Image.open(IMG / "70_closeup.png").convert("RGB"))[200:990, 280:980]
plt.rcParams["font.family"] = "DejaVu Sans"
W, H = 8.6, 5.0                       # each panel 0.886 wide per 1 tall, 0.88 of the height, small gap
fig = plt.figure(figsize=(W, H), dpi=200)
ph = 0.88
pw = 0.886 * ph * H / W
gap = (1 - 2 * pw) / 3
for i, (im, title) in enumerate(((over, "PiPER arm (simulation)"), (close, "Camera mount (orange) + U20CAM"))):
    ax = fig.add_axes([gap + i * (pw + gap), 0.0, pw, ph])
    ax.imshow(im); ax.axis("off")
    ax.set_title(title, fontsize=15, pad=6)
fig.savefig(OUT, dpi=200, facecolor="white")
print(OUT)
