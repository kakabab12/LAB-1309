"""Picture of the 11-20 weekly report: live teleop screen with a hand | 3D hand from two cameras.

Inputs (in the report folder):
  그림_손_조종화면.png   live screen of teleop.py --cam 0 --cam2 2 with the real PiPER connected (11/18 17:02 lab, 560x270,
                        saved by teleop.py --status): camera 1 with the right hand and its 21 keypoints on the A4
                        calibration grid, camera 2 in the corner, robot panel "REAL ARM". The status text lines at the
                        top were cut off when saving this file (that moment's line was from before the two-view hand pairing fix);
                        the uncut screen is not uploaded
  손3D_1118.json         right hand triangulated from the two cameras (11/18 16:51 lab, 2 views agree within 1.2 px)
Output: 그림_주간보고_1120.png (next to this folder)
11/19: the poster preview was dropped from the picture on request (the poster stays in the Previous text)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
R = HERE.parent
REPO = R.parent.parent
LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13),
         (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]

screen = mpimg.imread(R / "그림_손_조종화면.png")               # status text lines at the top already cut off
hand = json.load(open(R / "손3D_1118.json", encoding="utf-8"))

fig = plt.figure(figsize=(9.6, 3.3), dpi=200)
gs = fig.add_gridspec(1, 2, width_ratios=[7.2, 2.4], wspace=0.04, left=0.005, right=0.975, top=0.9, bottom=0.02)

ax = fig.add_subplot(gs[0])
ax.imshow(screen)
ax.set_title("Hand keypoints from two webcams driving the real PiPER (live screen)", fontsize=8.5)
for x, y, txt in ((0.18, 0.86, "camera 1"), (0.53, 0.47, "camera 2"), (0.80, 0.86, "robot (real arm)")):
    ax.text(x, y, txt, transform=ax.transAxes, ha="center", va="center", fontsize=6.5, color="white",
            bbox=dict(boxstyle="round,pad=0.25", fc="#1f77b4", ec="none", alpha=0.85))
ax.axis("off")

ax = fig.add_subplot(gs[1], projection="3d")
X = [[v * 100 for v in p] for p in hand["tab21_m"]]
for a, b in LINKS:
    ax.plot([X[a][0], X[b][0]], [X[a][1], X[b][1]], [X[a][2], X[b][2]], color="#2ca02c", lw=1.4)
ax.scatter([p[0] for p in X], [p[1] for p in X], [p[2] for p in X], s=5, color="#2ca02c")
ax.scatter([X[4][0], X[8][0]], [X[4][1], X[8][1]], [X[4][2], X[8][2]], s=14, color="#d62728", zorder=5)
px, py = 21.0 / 2, 29.7 / 2
ax.plot([-px, px, px, -px, -px], [-py, -py, py, py, -py], [0, 0, 0, 0, 0], color="#1f77b4", lw=1.0)
ax.plot([-4, 4], [0, 0], [0, 0], color="k", lw=0.8)
ax.plot([0, 0], [-4, 4], [0, 0], color="k", lw=0.8)
pin = [v * 100 for v in hand["pinch_m"]]
ax.plot([pin[0], pin[0]], [pin[1], pin[1]], [0, pin[2]], color="#d62728", lw=0.8, ls="--")
ax.text(pin[0], pin[1] + 2, pin[2] / 2, f"{pin[2]:.0f} cm", color="#d62728", fontsize=6.5)
ax.set_xlim(-12, 12)
ax.set_ylim(-16, 16)
ax.set_zlim(0, 20)
ax.set_box_aspect((24, 32, 20))
ax.view_init(elev=22, azim=-55)
ax.set_xlabel("forward (cm)", fontsize=6, labelpad=-6)
ax.set_ylabel("left (cm)", fontsize=6, labelpad=-6)
ax.set_zlabel("up (cm)", fontsize=6, labelpad=-7)
ax.set_xticks([-10, 0, 10])
ax.set_yticks([-15, 0, 15])
ax.set_zticks([0, 10, 20])
ax.tick_params(labelsize=5, pad=-2)
ax.set_title("3D hand from 2 cameras (triangulated)", fontsize=8.5)

out = R / "그림_주간보고_1120.png"
fig.savefig(out)
print("wrote", out)
