#!/usr/bin/env python
"""outputs/pose/*.json → 자세 민감도 표(markdown) + 그래프(PNG)."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC, REPORT = Path("outputs/pose"), Path("outputs/report")
TASKS = {7: "turn on the stove", 1: "put the bowl on the stove", 5: "push the plate to the front of the stove"}
COLORS = {7: "#2a78d6", 1: "#eb6834", 5: "#1baf7a"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"
plt.rcParams["font.family"] = ["NanumBarunGothic", "NanumGothic", "DejaVu Sans"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load():
    data = defaultdict(dict)  # (task, axis) -> {value: (k, n)}
    for f in SRC.glob("T*.json"):
        d = json.load(open(f))
        a = d["args"]
        eps = [e for e in d["episodes"] if e["reached_start_pose"]]
        k, n = sum(e["success"] for e in eps), len(eps)
        axis, val = ("yaw", a["yaw_deg"]) if a["yaw_deg"] else ("offset", a["offset_cm"])
        data[(a["task"], axis)][val] = (k, n)
        if a["yaw_deg"] == 0 and a["offset_cm"] == 0:  # 기준점은 두 축 모두에 사용
            data[(a["task"], "yaw")][0] = (k, n)
    return data


def plot(data, axis, xs, xlabel, title, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for t, lang in TASKS.items():
        pts = data[(t, axis)]
        ks = np.array([pts[x][0] for x in xs], float)
        ns = np.array([pts[x][1] for x in xs], float)
        p = ks / ns * 100
        lo, hi = zip(*[wilson(k, n) for k, n in zip(ks, ns)])
        ax.fill_between(xs, np.array(lo) * 100, np.array(hi) * 100, color=COLORS[t], alpha=0.12, lw=0)
        ax.plot(xs, p, color=COLORS[t], lw=2, marker="o", ms=6, label=f"{t}: {lang}")
    ax.set_xticks(xs)
    ax.set_ylim(-3, 105)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel("성공률 (%)", color=INK2)
    ax.set_title(title, color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left", bbox_to_anchor=(0, -0.42), ncol=1)
    fig.text(0.01, 0.005, "태스크당 10 에피소드, 띠 = Wilson 95% 신뢰구간", color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    data = load()
    offs, yaws = [0, 2, 4, 6, 8, 10], [0, 10, 20, 30]
    plot(data, "offset", offs, "초기 자세에서 끝단을 옮긴 거리 (cm)",
         "시작 위치를 옮기면 성공률이 어떻게 되나", REPORT / "pose_offset.png")
    plot(data, "yaw", yaws, "손목 회전 (도)",
         "손목을 돌린 채 시작하면 성공률이 어떻게 되나", REPORT / "pose_yaw.png")

    md = ["### 위치 오프셋 (손목 회전 0°)\n", "| 태스크 | " + " | ".join(f"{x}cm" for x in offs) + " |",
          "|---|" + "---|" * len(offs)]
    for t, lang in TASKS.items():
        md.append(f"| {t} {lang} | " + " | ".join(
            f"{100 * data[(t, 'offset')][x][0] / data[(t, 'offset')][x][1]:.0f}%" for x in offs) + " |")
    pooled = [(sum(data[(t, 'offset')][x][0] for t in TASKS), sum(data[(t, 'offset')][x][1] for t in TASKS))
              for x in offs]
    md.append("| **합계** | " + " | ".join(f"**{100 * k / n:.0f}%**" for k, n in pooled) + " |")

    md += ["\n### 손목 회전 (위치 오프셋 0cm)\n", "| 태스크 | " + " | ".join(f"{x}°" for x in yaws) + " |",
           "|---|" + "---|" * len(yaws)]
    for t, lang in TASKS.items():
        md.append(f"| {t} {lang} | " + " | ".join(
            f"{100 * data[(t, 'yaw')][x][0] / data[(t, 'yaw')][x][1]:.0f}%" for x in yaws) + " |")
    pooledy = [(sum(data[(t, 'yaw')][x][0] for t in TASKS), sum(data[(t, 'yaw')][x][1] for t in TASKS))
               for x in yaws]
    md.append("| **합계** | " + " | ".join(f"**{100 * k / n:.0f}%**" for k, n in pooledy) + " |")

    (REPORT / "pose_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
