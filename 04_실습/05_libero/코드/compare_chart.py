#!/usr/bin/env python
"""
여러 모델의 28개 항목 성공률을 한 그림에 나란히 (2026-10-02)

쓰는 법
  python compare_chart.py --png outputs/media/compare.png \
      --model "v6b 지연" outputs/v6blat_forget outputs/v6blat_switch \
      --model "v6c 지연" outputs/v6clat_forget outputs/v6clat_switch
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import scoreboard as sb

FP = "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"
font_manager.fontManager.addfont(FP)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FP).get_name()
PALETTE = ["#9aa5b1", "#2a78d6", "#eb6834", "#1baf7a", "#9650c8"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", nargs=3, action="append", metavar=("NAME", "FORGET", "SWITCH"), required=True)
    p.add_argument("--png", required=True)
    p.add_argument("--title", default="항목별 성공률 (실제 로봇 조건: 지연 0.56초)")
    a = p.parse_args()
    data = [(name, sb.collect([f], [s])) for name, f, s in a.model]
    keys = [r[1] for r in data[0][1]]
    groups = [r[0] for r in data[0][1]]
    m = len(data)
    w = 0.8 / m
    fig, ax = plt.subplots(figsize=(15, 5))
    for j, (name, rows) in enumerate(data):
        xs, vs = [], []
        for i, r in enumerate(rows):
            if r[4] > 0:
                xs.append(i - 0.4 + w * (j + 0.5))
                vs.append(100 * r[3] / r[4])
        ok = sum(v >= 85 for v in vs)
        ax.bar(xs, vs, width=w * 0.92, color=PALETTE[j % len(PALETTE)], label=f"{name} (85% 이상 {ok}/{len(vs)})")
        for x, v in zip(xs, vs):            # 0% 는 막대가 안 보이므로 표시 (안 잰 항목과 구분)
            if v == 0:
                ax.plot([x - w * 0.4, x + w * 0.4], [0.8, 0.8], color=PALETTE[j % len(PALETTE)], lw=2.5)
    ax.axhspan(85, 95, color="#1baf7a", alpha=0.10, lw=0)
    ax.axhline(85, color="#1baf7a", lw=1.2, ls="--")
    for b in (9.5, 21.5):
        ax.axvline(b, color="#ccc", lw=1)
    for x, t in ((4.5, "단독 10"), (15.5, "전환 12 (B 성공)"), (24.5, "재개 6")):
        ax.text(x, 104, t, ha="center", fontsize=10, color="#444")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=55, ha="right", fontsize=8)
    ax.set_ylim(0, 110)
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_ylabel("성공률 (%)")
    ax.set_title(a.title, loc="left", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.28), ncol=m, frameon=False, fontsize=9)
    fig.tight_layout()
    Path(a.png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.png, dpi=130)
    print("그림:", a.png)


if __name__ == "__main__":
    main()
