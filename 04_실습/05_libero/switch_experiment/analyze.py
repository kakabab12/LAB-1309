#!/usr/bin/env python
"""outputs/switch/*.json + traj/*.npz → 비교표(markdown), 에피소드 CSV, 그래프(PNG).

반응 시간: 같은 (쌍, 시점, 에피소드)의 strategy=none 궤적과 비교해, 전환 후 끝단 위치가
          2cm 이상 벌어지는 첫 스텝. (전환 전까지는 두 궤적이 완전히 동일)
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = Path("outputs/switch")
REPORT = Path("outputs/report")
STRATS = ["flush", "keep", "blend", "retreat"]
TIMINGS = ["step:15", "grasp:3", "grasp:20"]
TIMING_KO = {"step:15": "접근 중", "grasp:3": "잡은 직후", "grasp:20": "들고 이동 중"}
FAIL_TYPES = ["물체 낙하", "지시 무시", "얼어붙음", "시간 초과"]
# dataviz 기본 팔레트 categorical slot 1~4 (고정 순서), 기준(none)은 회색
COLORS = {"flush": "#2a78d6", "keep": "#eb6834", "blend": "#1baf7a", "retreat": "#eda100", "none": "#8a8984"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"
DIVERGE_M = 0.02


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load():
    eps = []
    for f in sorted(SRC.glob("A*_B*_*.json")):
        d = json.load(open(f))
        a = d["args"]
        for e in d["episodes"]:
            e.update(pair=f"{a['task_a']}→{a['task_b']}", task_a_id=a["task_a"], task_b_id=a["task_b"],
                     tag=f"A{a['task_a']}_B{a['task_b']}_{a['switch_at'].replace(':', '')}")
            eps.append(e)
    return eps


def add_reaction(eps):
    for e in eps:
        e["reaction_steps"] = None
        if e["strategy"] == "none" or not e.get("switched"):
            continue
        fn = SRC / "traj" / f"{e['tag']}_none_ep{e['episode']}.npz"
        fs = SRC / "traj" / f"{e['tag']}_{e['strategy']}_ep{e['episode']}.npz"
        if not (fn.exists() and fs.exists()):
            continue
        pn, ps = np.load(fn)["pos"], np.load(fs)["pos"]
        ts = e["a_steps_before_switch"]
        k = min(len(pn), len(ps))
        dev = np.linalg.norm(pn[:k] - ps[:k], axis=1)[ts:]
        hit = np.nonzero(dev > DIVERGE_M)[0]
        e["reaction_steps"] = int(hit[0]) + 1 if len(hit) else None


def rate_str(vals):
    vals = [bool(v) for v in vals if v is not None]
    if not vals:
        return "-"
    k, n = sum(vals), len(vals)
    return f"{100 * k / n:.0f}% ({k}/{n})"


def med(vals):
    vals = [v for v in vals if v is not None]
    return f"{np.median(vals):.1f}" if vals else "-"


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    eps = load()
    add_reaction(eps)
    sw = [e for e in eps if e.get("switched")]
    by = defaultdict(list)
    for e in sw:
        by[(e["strategy"], e["switch_at"])].append(e)

    md = []
    n_all = Counter((e["strategy"], e["switch_at"]) for e in eps)
    md.append("### 전략 × 전환 시점 (4개 태스크 쌍 합산)\n")
    rows = []
    for t in TIMINGS:
        ref = by[("none", t)]
        rows.append([TIMING_KO[t], "none(전환 안 함)", f"{len(ref)}/{n_all[('none', t)]}",
                     rate_str([e["holding_at_switch"] for e in ref]), "-", "-", "-",
                     rate_str([e["drop"] for e in ref]), "-", med([e["jerk"] for e in ref])])
        for s in STRATS:
            g = by[(s, t)]
            rows.append([TIMING_KO[t], s, f"{len(g)}/{n_all[(s, t)]}",
                         rate_str([e["holding_at_switch"] for e in g]),
                         rate_str([e.get("b_success") for e in g]),
                         rate_str([e.get("a_resume_success") for e in g]),
                         rate_str([e.get("both_success") for e in g]),
                         rate_str([e.get("drop") for e in g]),
                         med([e["reaction_steps"] for e in g]),
                         med([e["jerk"] for e in g])])
    md.append(table(rows, ["시점", "전략", "전환 성립", "잡은 상태", "B 성공", "A 재개", "둘 다",
                           "낙하", "반응(스텝, 중앙값)", "저크(중앙값)"]))

    md.append("\n\n### B 실패 유형 (전략별, 모든 시점 합산)\n")
    rows = []
    for s in STRATS:
        c = Counter(e["b_failure_type"] for e in sw if e["strategy"] == s and e.get("b_failure_type"))
        rows.append([s, sum(c.values())] + [c.get(ft, 0) for ft in FAIL_TYPES])
    md.append(table(rows, ["전략", "B 실패 수"] + FAIL_TYPES))

    md.append("\n\n### 태스크 쌍별 B 성공 / A 재개 (모든 시점 합산)\n")
    pairs = sorted({e["pair"] for e in sw})
    rows = []
    for p in pairs:
        g = [e for e in sw if e["pair"] == p]
        lang = next(e for e in g)
        row = [f"{p} ({lang['task_a']} → {lang['task_b']})"]
        for s in STRATS:
            gs = [e for e in g if e["strategy"] == s]
            row.append(f"{rate_str([e.get('b_success') for e in gs])} / "
                       f"{rate_str([e.get('a_resume_success') for e in gs])}")
        rows.append(row)
    md.append(table(rows, ["쌍"] + STRATS))

    (REPORT / "tables.md").write_text("\n".join(md) + "\n")
    keys = ["pair", "switch_at", "strategy", "episode", "switched", "grasp_step", "a_steps_before_switch",
            "holding_at_switch", "held_object", "a_success", "b_success", "b_failure_type", "a_completed_during_b",
            "a_resume_success", "both_success", "drop", "reaction_steps", "jerk", "wall_sec"]
    with open(REPORT / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(eps)

    plot_rates(by, "b_success", "전환 후 B 태스크 성공률", REPORT / "b_success.png")
    plot_rates(by, "a_resume_success", "B 이후 A 태스크 재개 성공률", REPORT / "a_resume.png")
    plot_example(sw)
    print("\n".join(md))
    print(f"\n저장: {REPORT}/tables.md, episodes.csv, *.png  (에피소드 {len(eps)}, 전환 성립 {len(sw)})")


def style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot_rates(by, key, title, path):
    plt.rcParams["font.family"] = ["NanumBarunGothic", "NanumGothic", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150, facecolor=SURFACE)
    style(ax)
    width, gap = 0.18, 0.02
    x = np.arange(len(TIMINGS))
    for i, s in enumerate(STRATS):
        ks, ns = [], []
        for t in TIMINGS:
            vals = [bool(e[key]) for e in by[(s, t)] if e.get(key) is not None]
            ks.append(sum(vals))
            ns.append(len(vals))
        p = np.array([k / n if n else np.nan for k, n in zip(ks, ns)])
        lo, hi = zip(*[wilson(k, n) for k, n in zip(ks, ns)])
        xs = x + (i - 1.5) * (width + gap)
        ax.bar(xs, p * 100, width, color=COLORS[s], label=s, zorder=2)
        ax.errorbar(xs, p * 100, yerr=[(p - np.array(lo)) * 100, (np.array(hi) - p) * 100], fmt="none",
                    ecolor=INK2, elinewidth=1, capsize=2, zorder=3)
    ax.set_xticks(x, [TIMING_KO[t] for t in TIMINGS], color=INK)
    ax.set_ylim(0, 105)
    ax.set_ylabel("성공률 (%)", color=INK2)
    ax.set_title(title, color=INK, loc="left", fontsize=12)
    ax.legend(ncol=4, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.1), labelcolor=INK)
    fig.text(0.01, 0.005, "4개 태스크 쌍 × 10 에피소드, 오차막대 = Wilson 95% 신뢰구간", color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_example(sw):
    """잡은 직후 전환: 전략별 끝단 높이(z) 궤적 예시 (8→7, 에피소드 0)."""
    plt.rcParams["font.family"] = ["NanumBarunGothic", "NanumGothic", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150, facecolor=SURFACE)
    style(ax)
    ts = None
    for s in ["none"] + STRATS:
        f = SRC / "traj" / f"A8_B7_grasp3_{s}_ep0.npz"
        if not f.exists():
            continue
        pos = np.load(f)["pos"][:260]
        e = next((e for e in sw if e["tag"] == "A8_B7_grasp3" and e["strategy"] == s and e["episode"] == 0), None)
        if e:
            ts = e["a_steps_before_switch"]
        ax.plot(pos[:, 2], color=COLORS[s], lw=2, label=s)
    if ts:
        ax.axvline(ts, color=INK2, ls="--", lw=1)
        ax.text(ts + 2, ax.get_ylim()[1], "지시 전환", color=INK2, va="top", fontsize=9)
    ax.set_xlabel("스텝 (20Hz)", color=INK2)
    ax.set_ylabel("끝단 높이 z (m)", color=INK2)
    ax.set_title("잡은 직후 전환 시 끝단 높이 궤적 (bowl→plate 도중 turn on the stove, 에피소드 0)", color=INK,
                 loc="left", fontsize=11)
    ax.legend(ncol=5, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.18), labelcolor=INK)
    fig.tight_layout()
    fig.savefig(REPORT / "traj_example.png", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
