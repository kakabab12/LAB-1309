#!/usr/bin/env python
"""여러 모양·N개 장면 성적표 (10/24): world_eval 결과(json)들을 조건별로 — 일 개수 k·끼어들기 방식, 옮긴 물체 모양, 장면 물체 수.
   '전부' = 끝났을 때 k 개 일이 모두 되어 있음 (앞 일이 무너지지 않음 포함). 3cm(LIBERO On) / 판 안 둘 다.
   python piper_sim/world_board.py outputs/world_w1 --name "w1 단독" --md outputs/piper/world_board_w1.md --png outputs/media/world_board_w1.png"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    p.add_argument("--name", default="")
    p.add_argument("--md", default=None)
    p.add_argument("--png", default=None)
    a = p.parse_args()
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import piper_sim.world as W
    eps = []
    for d in a.dirs:
        for f in sorted(glob.glob(f"{d}/*.json")):
            for e in json.load(open(f)):
                e["file"] = os.path.basename(f)
                eps.append(e)
    rows = []
    by = defaultdict(list)
    for e in eps:
        by[(e["k"], e["nested"])].append(e)
    lines = [f"### 여러 모양·N개 장면 성적표 {a.name}", "", "| 일 개수 | 방식 | 장면 | 전부 3cm | 95% 구간 | 전부 판 안 | 일별 3cm |", "|---|---|---|---|---|---|---|"]
    for (k, nested), es in sorted(by.items()):
        n = len(es)
        ok = sum(e["all_ok"] for e in es)
        lo, hi = wilson(ok, n)
        loose = sum(e["all_loose"] for e in es)
        t_ok = sum(t["ok"] for e in es for t in e["tasks"])
        t_n = sum(len(e["tasks"]) for e in es)
        lines.append(f"| {k} | {'끼어들기 겹침' if nested else '차례로'} | {n} | {ok}/{n} ({100 * ok / n:.0f}%) | {100 * lo:.0f}~{100 * hi:.0f}% | "
                     f"{loose}/{n} ({100 * loose / n:.0f}%) | {t_ok}/{t_n} ({100 * t_ok / max(t_n, 1):.0f}%) |")
        rows.append((k, nested, ok, n, loose, t_ok, t_n))
    # 모양별 (옮긴 물체), 물체 수별
    shape = defaultdict(lambda: [0, 0])
    nobj = defaultdict(lambda: [0, 0])
    for e in eps:
        for t in e["tasks"]:
            m = None
            for sh, s in W.CFG["shapes"].items():
                if f" {s['en']} on the" in t["task"]:
                    m = sh
            if m:
                shape[m][0] += t["ok"]
                shape[m][1] += 1
        b = f"{(e['n_objects'] - 1) // 4 * 4 + 1}~{(e['n_objects'] - 1) // 4 * 4 + 4}"
        nobj[b][0] += e["all_ok"]
        nobj[b][1] += 1
    lines += ["", "| 옮긴 물체 모양 | 일별 3cm |", "|---|---|"]
    lines += [f"| {W.CFG['shapes'][s]['ko'][0]} | {v[0]}/{v[1]} ({100 * v[0] / max(v[1], 1):.0f}%) |" for s, v in sorted(shape.items())]
    lines += ["", "| 장면 물체 수 | 전부 3cm |", "|---|---|"]
    lines += [f"| {b} | {v[0]}/{v[1]} ({100 * v[0] / max(v[1], 1):.0f}%) |" for b, v in sorted(nobj.items())]
    out = "\n".join(lines)
    print(out)
    if a.md:
        open(a.md, "w").write(out + "\n")
    if a.png and rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        font_manager.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
        plt.rcParams["font.family"] = "NanumGothic"
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for nested, col, lab in ((True, "#2e6fbf", "끼어들기 겹침"), (False, "#9aa7b8", "차례로")):
            rs = [r for r in rows if r[1] == nested]
            if not rs:
                continue
            ks = [r[0] for r in rs]
            v = [100 * r[2] / r[3] for r in rs]
            ci = [wilson(r[2], r[3]) for r in rs]
            ax.errorbar(ks, v, yerr=[[x - 100 * c[0] for x, c in zip(v, ci)], [100 * c[1] - x for x, c in zip(v, ci)]],
                        color=col, marker="o", lw=2, capsize=4, label=lab)
        ax.axhline(95, color="#c0392b", lw=1, ls="--")
        ax.text(ax.get_xlim()[0], 96, " 목표 95%", color="#c0392b", fontsize=9, va="bottom")
        ax.set_xlabel("일 개수 k (한 번 실행에 옮기는 물체 수)")
        ax.set_ylabel("k 개 모두 성공 (%)")
        ax.set_ylim(0, 105)
        ax.set_title(f"여러 모양·N개 장면 {a.name}", loc="left", fontsize=12)
        ax.legend(frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#e6e6e6")
        fig.tight_layout()
        fig.savefig(a.png, dpi=140)
        print("그림:", a.png)


if __name__ == "__main__":
    main()
