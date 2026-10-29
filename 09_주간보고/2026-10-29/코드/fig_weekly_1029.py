"""Weekly report 10-29 picture (simple): a 13-object scene (left) + one-task success over 150 scenes, model alone vs model + small fix (right).
Plain words only on the slide (no VLA, digital twin ...). FRAME: many_23.jpg, a frame of outputs/media/world_w3r_many_s1006.mp4"""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
OUT, FRAME, RES = sys.argv[1], sys.argv[2], sys.argv[3]
img = np.asarray(Image.open(FRAME))[80:975, 230:1520]
plt.rcParams["font.family"] = "DejaVu Sans"
fig = plt.figure(figsize=(10, 3.85), dpi=200)
ax0 = fig.add_axes([0.0, 0.0, 0.545, 1.0])
ax0.imshow(img); ax0.axis("off")
ax = fig.add_axes([0.66, 0.14, 0.32, 0.74])
v = [100 * np.mean([e["all_ok"] for c in ("default", "random", "many") for e in json.load(open(f"{RES}/world_{m}_{c}/k1n.json"))]) for m in ("w7", "w7r")]
ax.bar([0, 1], v, 0.6, color=["#9aa5b1", "#2a6f97"], zorder=2)
for x, y in zip([0, 1], v):
    ax.text(x, y + 2, f"{y:.0f}%", ha="center", va="bottom", fontsize=18, color="#222")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Model alone", "Model +\nsmall fix"], fontsize=13)
ax.set_ylim(0, 70); ax.set_yticks([]); ax.spines[["top", "right", "left"]].set_visible(False)
ax.set_title("Success rate", fontsize=14)
fig.savefig(OUT, dpi=200)
print(OUT, v)
