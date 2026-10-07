"""Figure 3: stacking success vs number of demonstrations per stage (simulation).

Reads results/eval/v5_n{100,200,500,1000}/results.json (final design: fixed wall + back-off), v5_dart{200,1000},
and the v2 chained points (robot-facing wall) as a grey reference.
Lines: per-stage success (1층, 옆, 2층) and chained 3-stage success. Colour + marker + dash so the
figure survives greyscale printing; series are direct-labelled at the line ends.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 7, "axes.linewidth": 0.6})

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [  # (label, colour, marker, dash)
    ("1층", "#2a78d6", "o", "-"),
    ("옆", "#eb6834", "s", "-"),
    ("2층", "#1baf7a", "^", "-"),
    ("연속 3단계", "#eda100", "D", "--"),
]


def load(name):
    p = ROOT / "results" / "eval" / name / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


def main(out: Path, demo: bool = False) -> None:
    ns = [100, 200, 500, 1000]
    data = {n: load(f"v5_n{n}") for n in ns}
    if demo:
        import random
        random.seed(0)
        data = {n: {"per_stage": {str(s): {"success_rate": min(1, 0.5 + 0.1 * i + 0.05 * s * random.random())}
                                  for s in (1, 2, 3)},
                    "chained": {"cumulative_success": [0.5, 0.4, 0.3 + 0.1 * i]}} for i, n in enumerate(ns)}
    xs = [n for n in ns if data[n]]
    if len(xs) < 2:
        print("not enough results yet")
        return
    fig, ax = plt.subplots(figsize=(3.25, 2.05), dpi=300)
    ends = []
    for k, (label, col, mk, ls) in enumerate(SERIES):
        if k < 3:
            ys = [100 * data[n]["per_stage"][str(k + 1)]["success_rate"] for n in xs]
        else:
            ys = [100 * data[n]["chained"]["cumulative_success"][2] for n in xs]
        ax.plot(xs, ys, color=col, lw=1.4, ls=ls, marker=mk, ms=3.6, mec="white", mew=0.6, zorder=3)
        ends.append([ys[-1], label, col])
    if not demo:  # v2 reference: same 9 mm clearance, robot-facing wall (switches near +8 deg)
        ref = [(n, load(f"n{n}")) for n in (100, 200)]
        ref = [(n, d) for n, d in ref if d and "chained" in d]
        if ref:
            ax.plot([n for n, _ in ref], [100 * d["chained"]["cumulative_success"][2] for _, d in ref], color="#9a9994",
                    lw=1.0, ls=":", marker="D", ms=3, mec="white", mew=0.5, zorder=2)
            n, d = ref[-1]
            ax.annotate("로봇 쪽 벽 시연 (연속)", (n, 100 * d["chained"]["cumulative_success"][2]), xytext=(4, -8),
                        textcoords="offset points", fontsize=5.8, color=INK2)
    darts = [] if demo else [(n, load(f"v5_dart{n}")) for n in (200, 1000)]
    darts = [(n, d) for n, d in darts if d and "chained" in d]
    if darts:
        dx = [n for n, _ in darts]
        dy = [100 * d["chained"]["cumulative_success"][2] for _, d in darts]
        ax.plot(dx, dy, marker="*", ms=8, color="#4a3aa7", mec="white", mew=0.5, zorder=4, lw=1.0, ls=":")
        ax.annotate("DART (연속)", (dx[-1], dy[-1]), xytext=(6, 6), textcoords="offset points", fontsize=6.2,
                    color=INK2, va="center")
    # direct labels at the right end, nudged apart
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < 6:
            ends[i][0] = ends[i - 1][0] + 6
    for y, label, col in ends:
        ax.text(xs[-1] * 1.08, y, label, fontsize=6.4, color=INK, va="center")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlim(85, 1000 * 1.75)
    ax.set_ylim(0, 105)
    ax.set_xlabel("단계별 시연 수", color=INK2)
    ax.set_ylabel("성공률 (%)", color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.minorticks_off()
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#9a9994")
    ax.tick_params(colors=INK2, length=2.5, width=0.6)
    fig.tight_layout(pad=0.3)
    fig.savefig(out, dpi=300)
    print("saved", out)


if __name__ == "__main__":
    demo = "--demo" in sys.argv
    main(Path("/tmp/claude-1000/-home-user-ACT/a30174b6-8651-4d46-9293-5a1be058a6f5/scratchpad/fig3_demo.png") if demo
         else ROOT / "paper" / "fig3_scaling.png", demo)
