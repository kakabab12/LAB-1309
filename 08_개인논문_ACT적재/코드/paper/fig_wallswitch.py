"""Why the v2 policy failed: outcome of every per-stage trial vs the bin's yaw on the scale.

The v1-v3 expert grasps the wall facing the robot, which flips from the front wall to the side wall near
yaw +10 deg (the exact angle depends on the bin position). Demonstrations become bimodal there and almost
all ACT failures sit in that band. One row per stage; colour = wall the expert would grasp, x = failure.
usage: python paper/fig_wallswitch.py [run=n100]
"""

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stack_env import sample_scene  # noqa: E402

font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 7, "axes.linewidth": 0.6})
INK2, GRID = "#52514e", "#e4e3df"
FRONT, SIDE = "#2a78d6", "#eb6834"


def facing_k(center_xy, yaw):
    to = -np.asarray(center_xy[:2]) / np.linalg.norm(center_xy[:2])
    d = [np.array([math.cos(yaw + k * math.pi / 2), math.sin(yaw + k * math.pi / 2)]) @ to for k in range(4)]
    return int(np.argmax(d))


def main(run="n100"):
    r = json.loads((ROOT / "results" / "eval" / run / "results.json").read_text())
    fig, ax = plt.subplots(figsize=(3.3, 1.75), dpi=300)
    rows, stats = [], {}
    for si, st in enumerate(("1", "2", "3")):
        for t in r["per_stage"][st]["trials"]:
            spec = sample_scene(int(st), np.random.default_rng(500000 + 1000 * int(st) + t["trial"]))
            yaw = math.degrees(spec.new_bin_yaw)
            k = facing_k(t["bin0"], spec.new_bin_yaw)
            rows.append((si, yaw, k, t["success"]))
    yaws = np.array([x[1] for x in rows])
    fails = np.array([not x[3] for x in rows])
    hi = yaws > 7.5  # band around the wall switch
    stats = {"trials": len(rows), "fail_total": int(fails.sum()),
             "band_trials": int(hi.sum()), "band_fails": int((fails & hi).sum()),
             "rest_trials": int((~hi).sum()), "rest_fails": int((fails & ~hi).sum())}
    lo = min(x[1] for x in rows if x[2] == 1)
    ax.axvspan(lo, 20.5, color="#f3f2ee", zorder=0)
    ax.text(lo + 0.3, 2.62, "전문가가 잡는 벽이 바뀌는 구간", fontsize=5.6, color=INK2, va="center")
    for si, yaw, k, ok in rows:
        col = FRONT if k == 2 else SIDE
        y = 2 - si + (0.12 if ok else -0.12)
        if ok:
            ax.scatter(yaw, y, s=7, facecolor="white", edgecolor=col, lw=0.6, zorder=3)
        else:
            ax.scatter(yaw, y, s=14, marker="x", color=col, lw=0.9, zorder=4)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(["1층", "옆", "2층"])
    ax.set_ylim(-0.5, 2.85)
    ax.set_xlim(-20.5, 20.5)
    ax.set_xlabel("저울 위 통의 회전각 (°)", color=INK2)
    ax.scatter([], [], s=7, facecolor="white", edgecolor=INK2, lw=0.6, label="성공")
    ax.scatter([], [], s=14, marker="x", color=INK2, lw=0.9, label="실패")
    ax.scatter([], [], s=9, marker="s", color=FRONT, label="전문가: 앞벽 파지")
    ax.scatter([], [], s=9, marker="s", color=SIDE, label="전문가: 옆벽 파지")
    fig.legend(frameon=False, fontsize=5.8, loc="upper center", ncol=4, handletextpad=0.2, columnspacing=0.9)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#9a9994")
    ax.tick_params(colors=INK2, length=2.5, width=0.6)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.5)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.9))
    out = ROOT / "paper" / f"fig_wallswitch_{run}.png"
    fig.savefig(out, dpi=300)
    (ROOT / "results" / f"wallswitch_{run}.json").write_text(json.dumps(stats, indent=1))
    print("saved", out, stats)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["n100"]))
