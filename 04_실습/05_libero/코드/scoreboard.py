#!/usr/bin/env python
"""
성적표 — 28개 항목을 한 장에 (2026-10-02)

  단독 10개 (T0~T9)      : 그 태스크만 시켰을 때 성공
  전환 12쌍 (A→B)        : A 를 하다 물체를 쥔 직후 B 지시 → B 성공
  재개 6쌍 (A→B→A)       : B 를 마친 뒤 내려놓은 물체를 다시 집어 A 까지 성공 (원래 끝나 있던 경우는 뺀다)

목표: 항목마다 85% 이상 (95% 넘는 것은 일부러 낮추지 않는다)

쓰는 법
  python scoreboard.py --forget outputs/v6blat_forget --switch outputs/v6blat_switch --name "v6b 지연11" \
      --png outputs/media/score_v6b_lat.png --md outputs/v6/score_v6b_lat.md
"""
import argparse
import glob
import json
import math
from pathlib import Path

PAIRS = ["8:0", "8:3", "8:5", "8:7", "8:9", "4:5", "4:9", "1:7", "8:1", "8:4", "1:8", "4:1"]
RESUME = ["8:5", "8:7", "8:9", "4:5", "4:9", "1:7"]
SHORT = {0: "가운데 서랍 열기", 1: "그릇→스토브", 2: "와인→캐비닛 위", 3: "위 서랍 열고 그릇 넣기", 4: "그릇→캐비닛 위",
         5: "접시 밀기", 6: "치즈→그릇", 7: "스토브 켜기", 8: "그릇→접시", 9: "와인→선반"}
BLOCK_SHORT = {0: "파랑→흰 판", 1: "빨강→주황 판", 2: "노랑 막대→보라 판", 3: "빨강→파랑 위", 4: "빨강→회색 판",
               5: "초록→흰 판", 6: "초록→빨강 위", 7: "노랑 막대→흰 판", 8: "빨강→보라 판", 9: "초록→파랑 위"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def episodes(dirs, pat):
    out = {}
    for d in dirs:
        for f in glob.glob(f"{d}/{pat}"):
            for e in json.load(open(f)).get("episodes", []):
                out[e["episode"]] = e
    return list(out.values())


def collect(forget, switch):
    rows = []
    # LOOSE: '판 안' 기준 (10/5, 사용자 결정: 3cm 기준과 둘 다 보고). 블록 장면이 아닌 기록에는 값이 없어 3cm 와 같게 센다
    sfx = "_loose" if LOOSE else ""
    for t in range(10):
        es = episodes(forget, f"A{t}_Bnone_*.json")
        rows.append(("단독", f"T{t}", SHORT[t], sum(bool(e.get("a_success" + sfx, e.get("a_success"))) for e in es), len(es)))
    # 10/5: 분모는 전체 시도 — 전환 전에 물체를 못 쥐어 B 지시를 못 받은 시도도 실패로 센다
    #   (예전에는 전환된 시도만 셌다. 쥠 판정이 엄격해진 뒤로는 그러면 부풀어 보인다)
    for p in PAIRS:
        a, b = p.split(":")
        es = episodes(switch, f"A{a}_B{b}_*.json")
        rows.append(("전환", f"A{a}→B{b}", f"{SHORT[int(a)]} → {SHORT[int(b)]}",
                     sum(bool(e.get("switched")) and bool(e.get("b_success" + sfx, e.get("b_success"))) for e in es), len(es)))
    for p in RESUME:
        a, b = p.split(":")
        es = episodes(switch, f"A{a}_B{b}_*.json")
        rows.append(("재개", f"A{a}→B{b}→A{a}", f"B 뒤 {SHORT[int(a)]}",
                     sum(bool(e.get("switched")) and bool(e.get("a_resume_genuine" + sfx, e.get("a_resume_genuine"))) for e in es), len(es)))
    return rows


LOOSE = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--forget", nargs="*", default=[])
    p.add_argument("--switch", nargs="*", default=[])
    p.add_argument("--name", default="")
    p.add_argument("--png", default=None)
    p.add_argument("--md", default=None)
    p.add_argument("--loose", action="store_true", help="'판 안' 기준 (블록 장면, 10/5)")
    p.add_argument("--blocks", action="store_true", help="블록 장면 과제 이름으로")
    a = p.parse_args()
    global LOOSE
    LOOSE = a.loose
    if a.blocks:
        SHORT.update(BLOCK_SHORT)
    rows = collect(a.forget, a.switch)
    lines = [f"### 성적표 {a.name}", "", "| 구분 | 항목 | 내용 | 성공 | 성공률 | 95% 구간 | 95% 이상 |", "|---|---|---|---|---|---|---|"]
    ok = tot = 0
    for g, k, desc, s, n in rows:
        if n == 0:
            lines.append(f"| {g} | {k} | {desc} | - | - | - | - |")
            continue
        lo, hi = wilson(s, n)
        hit = s / n >= 0.95
        ok += hit
        tot += 1
        lines.append(f"| {g} | {k} | {desc} | {s}/{n} | {100 * s / n:.0f}% | {100 * lo:.0f}~{100 * hi:.0f}% | {'달성' if hit else ''} |")
    lines += ["", f"95% 이상: {ok} / {tot} 항목" + (" (판 안 기준)" if LOOSE else "")]
    txt = "\n".join(lines)
    print(txt)
    if a.md:
        Path(a.md).write_text(txt + "\n")
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        for fp in ["/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
            if Path(fp).exists():
                font_manager.fontManager.addfont(fp)
                plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
                break
        col = {"단독": "#2a78d6", "전환": "#eb6834", "재개": "#1baf7a"}
        R = [r for r in rows if r[4] > 0]
        fig, ax = plt.subplots(figsize=(13, 4.6))
        xs = range(len(R))
        vals = [100 * r[3] / r[4] for r in R]
        ax.bar(xs, vals, color=[col[r[0]] for r in R], width=0.72)
        for x, r, v in zip(xs, R, vals):
            ax.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color="#333")
        ax.axhspan(95, 100, color="#1baf7a", alpha=0.10, lw=0)
        ax.axhline(95, color="#1baf7a", lw=1.2, ls="--")
        ax.text(len(R) - 0.4, 96.5, "목표 95% 이상", ha="right", fontsize=9, color="#15925f")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([r[1] for r in R], rotation=55, ha="right", fontsize=8)
        ax.set_ylim(0, 108)
        ax.set_ylabel("성공률 (%)")
        n_ep = max(r[4] for r in R)
        ax.set_title(f"{a.name} — 95% 이상 {ok}/{tot} 항목 (항목당 {n_ep}회)", fontsize=11, loc="left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=c, label=g) for g, c in col.items()], loc="upper left", frameon=False, ncol=3,
                  fontsize=9, bbox_to_anchor=(0, 1.0))
        fig.tight_layout()
        Path(a.png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.png, dpi=130)
        print("그림:", a.png)


if __name__ == "__main__":
    main()
