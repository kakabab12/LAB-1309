#!/usr/bin/env python
"""
"얼마나 가까이 가야 정책이 살아나는가" — 용량-반응 곡선

배경 (2026-09-18): 부분적으로 초기 자세에 가까워지는 것은 효과가 없었다.
  흔들기 4~10cm 이동 → 25~31% (변화 없음)
  rollback (초기위치 21cm) → 19% (더 나쁨)
  ret_pos (초기위치 0cm) → 64%,  retreat (0cm + 회전) → 89%

그래서 **초기 자세 쪽으로 f 만큼만** 되돌리고(위치·회전 모두) 성공률을 잰다.
  f = 0 은 release(8%), f = 1 은 retreat(89%) 와 같다.

읽는 법
  f = 0.5 에서 이미 높다  → **부분 복귀로 충분** = 초기 자세가 아니므로 제약을 지킬 수 있다
  f = 0.9 까지 낮다        → 사실상 완전 복귀가 필요 = 제약과 충돌 → 학습으로 정책을 바꿔야 한다
"""
import glob
import json
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["NanumBarunGothic", "NanumGothic", "DejaVu Sans"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"
# 이미 측정된 양 끝점 (f=0 은 release, f=1 은 retreat, grasp:3 조건)
KNOWN = {0.0: ("release", 0.11), 1.0: ("retreat", 0.92)}


def sign_test(w, l):
    k = w + l
    return min(1.0, 2 * sum(comb(k, i) for i in range(max(w, l), k + 1)) / 2 ** k) if k else 1.0


def load(pattern, key_fn, only_grasp3=True):
    rows = defaultdict(list)
    paired = defaultdict(dict)
    for f in glob.glob(pattern):
        d = json.load(open(f))
        a = d["args"]
        if only_grasp3 and a["switch_at"] != "grasp:3":
            continue
        k = key_fn(a)
        if k is None:
            continue
        for e in d["episodes"]:
            if not e.get("switched"):
                continue
            rows[k].append(e)
            paired[(a["task_a"], a["task_b"], e["episode"])][k] = e
    return rows, paired


def main():
    rows, paired = load("outputs/retpart/A*_B*_grasp3_ret_part_rf*.json",
                        lambda a: round(float(a["retreat_frac"]), 2))
    base, base_paired = load("outputs/switch/A*_B*_grasp3_flush.json",
                             lambda a: "flush" if (not a.get("latency_steps", 0)
                                                   and not a.get("switch_n_action_steps", 0)
                                                   and a.get("switch_noise_seed", -1) < 0
                                                   and a.get("dither_escape", "off") == "off") else None)
    for k, v in base_paired.items():
        paired[k].update(v)
    rows["flush"] = base["flush"]

    if not any(isinstance(k, float) for k in rows):
        print("outputs/retpart 에 결과가 없습니다."); return

    fracs = sorted(k for k in rows if isinstance(k, float))
    print(f"{'f':>6s}{'n':>5s}{'B 성공':>14s}{'A 재개':>12s}{'초기위치까지':>14s}"
          f"{'경로(cm)':>11s}{'우회':>7s}{'멈춤':>6s}")

    def show(label, k):
        r = rows.get(k)
        if not r:
            return None
        b = [bool(e.get("b_success")) for e in r]
        bs = [e for e in r if e.get("b_success")]
        res = [bool(e.get("a_resume_success")) for e in bs]
        hd = [e["home_dist_after_cm"] for e in r if e.get("home_dist_after_cm") is not None]
        path = [e["path_cm"] for e in r if e.get("path_cm") is not None]
        det = [e["detour"] for e in r if e.get("detour") is not None]
        st = [e["stops"] for e in r if e.get("stops") is not None]
        print(f"{label:>6s}{len(r):5d}{f'{100 * np.mean(b):.0f}% ({sum(b)}/{len(b)})':>14s}"
              f"{(f'{sum(res)}/{len(res)}' if res else '-'):>12s}"
              f"{(f'{np.median(hd):.1f}cm' if hd else '-'):>14s}"
              f"{(f'{np.median(path):.0f}' if path else '-'):>11s}"
              f"{(f'{np.median(det):.2f}' if det else '-'):>7s}"
              f"{(f'{np.median(st):.0f}' if st else '-'):>6s}")
        return {"n": len(r), "b": float(np.mean(b)),
                "resume": (float(np.mean(res)) if res else None),
                "home_cm": (float(np.median(hd)) if hd else None)}

    out = {}
    out["flush"] = show("flush", "flush")
    for f in fracs:
        out[str(f)] = show(f"{f:g}", f)

    print("\n짝 비교 (같은 초기상태, vs flush)")
    for f in fracs:
        w = l = 0
        for v in paired.values():
            if f in v and "flush" in v:
                x, y = bool(v[f].get("b_success")), bool(v["flush"].get("b_success"))
                w += x and not y
                l += y and not x
        print(f"  f={f:g}: f 만 {w:3d}, flush 만 {l:3d}, p = {sign_test(w, l):.3f}")

    # ---- 해석 ----
    print("\n== 해석")
    curve = [(f, out[str(f)]["b"]) for f in fracs if out.get(str(f))]
    if curve:
        half = [b for f, b in curve if abs(f - 0.5) < 1e-6]
        if half and half[0] >= 0.6:
            print(f"f=0.5 에서 이미 {100 * half[0]:.0f}% → **부분 복귀로 충분.** 초기 자세가 아니므로 제약을 지킬 수 있다")
            print("   → 이 동작을 LoRA 로 증류해 추론 때 스크립트를 없애는 것이 다음 단계")
        elif all(b < 0.5 for f, b in curve):
            print("모든 부분 복귀가 50% 미만 → **사실상 완전 복귀가 필요하다.** 제약과 정면 충돌")
            print("   → 학습으로 정책 자체를 낯선 자세에서도 작동하게 만들어야 한다")
        else:
            print("중간에서 올라간다 → 곡선의 무릎 지점을 찾아 그 값을 쓰면 된다")

    Path("outputs/report").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("outputs/report/retpart.json", "w"), ensure_ascii=False, indent=2)

    # ---- 그래프 ----
    xs = [0.0] + fracs + [1.0]
    ys = [KNOWN[0.0][1]] + [out[str(f)]["b"] for f in fracs] + [KNOWN[1.0][1]]
    fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=150, facecolor=SURF)
    ax.set_facecolor(SURF)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(INK2); ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.plot(xs, [100 * y for y in ys], color="#2a78d6", lw=2.5, marker="o", ms=6, zorder=3)
    if out.get("flush"):
        ax.axhline(100 * out["flush"]["b"], color="#8a8984", ls="--", lw=1.5)
        ax.text(0.02, 100 * out["flush"]["b"] + 2, f"그 자리 전환 {100 * out['flush']['b']:.0f}%",
                color=INK2, fontsize=9)
    ax.annotate("release", (0.0, 100 * ys[0]), textcoords="offset points", xytext=(6, -14),
                color=INK2, fontsize=9)
    ax.annotate("retreat (⛔ 제약 위반)", (1.0, 100 * ys[-1]), textcoords="offset points",
                xytext=(-120, -16), color=INK2, fontsize=9)
    ax.set_xlabel("초기 자세 쪽으로 되돌린 비율 f", color=INK2)
    ax.set_ylabel("B 성공률 (%)", color=INK2)
    ax.set_ylim(0, 100)
    ax.set_title("얼마나 가까이 가야 정책이 다시 작동하는가", color=INK, loc="left", fontsize=12)
    fig.text(0.01, 0.01, "4쌍 × 10회, 잡은 직후 전환. f=0 은 release, f=1 은 retreat 과 같다",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig("outputs/report/retpart.png", facecolor=SURF)
    print("저장: outputs/report/retpart.png, retpart.json")


if __name__ == "__main__":
    main()
