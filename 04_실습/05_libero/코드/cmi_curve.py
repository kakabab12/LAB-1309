#!/usr/bin/env python
"""
귀는 언제 닫히고 언제 열리는가 — CMI 시간 곡선

지금까지 안 것 (2026-09-21)
    에피소드 시작 0.719  →  잡고 3스텝 뒤 0.284  →  잡고 20스텝 뒤 0.761
  점 세 개뿐이라 **언제 닫히고 언제 열리는지** 모른다.

왜 곡선이 필요한가
  ① **기제를 확정한다.** 잡는 순간에 뚝 떨어졌다 서서히 회복되는지,
     아니면 애초에 3스텝 지점만 이상한지 구분된다
  ② **취약 구간의 길이**를 알 수 있다 — 논문에서 문제를 정의하는 숫자가 된다
  ③ **가장 단순한 기준선**이 생긴다: "귀가 열릴 때까지 N스텝 기다렸다 전환한다".
     이것이 우리 방법이 반드시 이겨야 할 상대다
     (단, 즉시성을 포기하는 것이라 같은 문제를 푸는 것은 아니다)

쓰는 법
  python cmi_curve.py [폴더]          (기본: outputs/steer_curve)
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["NanumBarunGothic", "NanumGothic", "DejaVu Sans"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"


def probe_step(name):
    """probe 이름을 '잡은 뒤 스텝 수' 로. start 는 잡기 전이라 None."""
    if name == "start":
        return None
    return int(name[5:]) if name.startswith("grasp") and name[5:].isdigit() else None


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "outputs/steer_curve"
    f = Path(src) / "steer.json"
    if not f.exists():
        print(f"{f} 가 없습니다.")
        return
    rows = json.load(open(f))["rows"]

    start = [r["cmi_nats"] for r in rows if r["probe"] == "start"]
    pts = {}
    for r in rows:
        s = probe_step(r["probe"])
        if s is not None:
            pts.setdefault(s, []).append(r["cmi_nats"])
    if not pts:
        print("grasp<N> 측정이 없습니다.")
        return
    xs = sorted(pts)

    print(f"== CMI 시간 곡선 ({src})\n")
    if start:
        print(f"{'잡기 전 (시작)':<18}{np.mean(start):9.4f}  n={len(start)}")
    print(f"{'잡은 뒤':<18}{'CMI':>9}{'표준편차':>10}{'n':>5}   시작 대비")
    base = np.mean(start) if start else None
    for s in xs:
        v = pts[s]
        rel = f"{100 * np.mean(v) / base:6.0f}%" if base else ""
        bar = "█" * max(1, int(round(20 * np.mean(v) / max(base or max(np.mean(pts[q]) for q in xs), 1e-9))))
        print(f"  {s:>3d} 스텝{'':<8}{np.mean(v):9.4f}{np.std(v):10.4f}{len(v):5d}   {rel}  {bar}")

    lo = min(xs, key=lambda s: np.mean(pts[s]))
    print(f"\n가장 낮은 지점: 잡은 뒤 **{lo}스텝** (CMI {np.mean(pts[lo]):.4f})")
    if base:
        # 시작값의 80% 를 회복한 첫 지점
        rec = [s for s in xs if s > lo and np.mean(pts[s]) >= 0.8 * base]
        if rec:
            print(f"시작값의 80% 를 회복하는 시점: 잡은 뒤 **{rec[0]}스텝** "
                  f"(20Hz 기준 {rec[0] / 20:.1f}초)")
            print(f"→ **취약 구간은 약 {rec[0] - lo}스텝 ({(rec[0] - lo) / 20:.1f}초)** 이다")
        else:
            print(f"측정 범위({max(xs)}스텝) 안에서는 시작값의 80% 를 회복하지 못한다")
            print("→ 더 뒤까지 재야 한다")

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150, facecolor=SURF)
    ax.set_facecolor(SURF)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ys = [np.mean(pts[s]) for s in xs]
    es = [np.std(pts[s]) / max(np.sqrt(len(pts[s])), 1) for s in xs]
    ax.errorbar(xs, ys, yerr=es, color="#eb6834", lw=2.5, marker="o", ms=5, capsize=3)
    if base:
        ax.axhline(base, color="#1baf7a", ls="--", lw=1.5)
        ax.text(max(xs) * 0.62, base * 1.02, f"잡기 전 {base:.2f}", color="#1baf7a", fontsize=9)
    ax.set_xlabel("물체를 잡은 뒤 지난 스텝 (20Hz)", color=INK2)
    ax.set_ylabel("CMI (지시문이 동작을 가르는 정도)", color=INK2)
    ax.set_ylim(bottom=0)
    ax.set_title("물체를 쥐면 지시문이 안 들린다 — 언제 닫히고 언제 열리나",
                 color=INK, loc="left", fontsize=12)
    fig.text(0.01, 0.01, "낮을수록 '무슨 말을 해도 같은 행동'. 전환 실험은 잡은 뒤 3스텝에서 한다",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    Path("outputs/report").mkdir(parents=True, exist_ok=True)
    fig.savefig("outputs/report/cmi_curve.png", facecolor=SURF)
    json.dump({"start": base, "curve": {str(s): float(np.mean(pts[s])) for s in xs}},
              open("outputs/report/cmi_curve.json", "w"), ensure_ascii=False, indent=2)
    print("\n저장: outputs/report/cmi_curve.png, cmi_curve.json")


if __name__ == "__main__":
    main()
