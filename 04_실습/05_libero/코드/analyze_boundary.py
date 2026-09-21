#!/usr/bin/env python
"""
데이터 수집 경계 — 큰 교란에서도 정책이 성공하는가

왜 재나
  ① **역커리큘럼 설계**: 자기 궤적으로 학습 데이터를 모으려면 그 교란에서 성공해야 한다.
     성공률이 0% 면 데이터를 못 모으고, 단계적으로 넓혀 가야 한다
  ② **전환 난이도의 분해**: 전환 시점은 초기 위치에서 27cm 떨어져 있다.
     깨끗한 장면에서 27cm 떨어졌을 때의 성공률과 비교하면,
     "전환의 어려움이 팔 위치만으로 설명되는가"를 알 수 있다

읽는 법
  깨끗한 장면 27cm ≈ 전환 성공률  → 전환의 어려움은 **팔 위치가 거의 전부**
  깨끗한 장면 27cm ≫ 전환 성공률  → 전환에는 **위치 말고 다른 어려움**이 더 있다
"""
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["NanumBarunGothic", "NanumGothic", "DejaVu Sans"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"
# 이미 측정된 값 (2026-09-16 자세 민감도, 태스크 7)
KNOWN_OFF = {0: 1.00, 4: 1.00, 8: 0.90}
KNOWN_YAW = {0: 1.00, 10: 0.90, 20: 0.60, 30: 0.00}


def load(src="outputs/pose_far"):
    rows = []
    for f in glob.glob(f"{src}/T*.json"):
        d = json.load(open(f))
        a = d["args"]
        rows.append({"task": a["task"], "off": float(a["offset_cm"]), "yaw": float(a["yaw_deg"]),
                     "rate": d["summary"]["success_rate"], "n": d["summary"].get("n_reached", 0)})
    return rows


def main():
    rows = load()
    if not rows:
        print("outputs/pose_far 에 결과가 없습니다."); return
    off_only = sorted([r for r in rows if r["yaw"] == 0], key=lambda r: r["off"])
    yaw_only = sorted([r for r in rows if r["off"] == 0], key=lambda r: r["yaw"])
    both = sorted([r for r in rows if r["off"] > 0 and r["yaw"] > 0], key=lambda r: (r["off"], r["yaw"]))

    print("== 위치 교란만 (태스크 7)")
    for cm, v in sorted(KNOWN_OFF.items()):
        print(f"  {cm:5.0f}cm  {100 * v:3.0f}%   (2026-09-16 측정)")
    for r in off_only:
        print(f"  {r['off']:5.0f}cm  {100 * r['rate']:3.0f}%   (n={r['n']})")

    print("\n== 손목 회전만 (태스크 7)")
    for d, v in sorted(KNOWN_YAW.items()):
        print(f"  {d:5.0f}도  {100 * v:3.0f}%   (2026-09-16 측정)")
    for r in yaw_only:
        print(f"  {r['yaw']:5.0f}도  {100 * r['rate']:3.0f}%   (n={r['n']})")

    if both:
        print("\n== 위치 + 회전 (전환 시점과 비슷한 조건)")
        for r in both:
            print(f"  {r['off']:3.0f}cm + {r['yaw']:3.0f}도  {100 * r['rate']:3.0f}%   (n={r['n']})")

    # ---- 전환 난이도 분해 ----
    curve = {**KNOWN_OFF, **{r["off"]: r["rate"] for r in off_only}}
    xs = sorted(curve)
    if len(xs) >= 2:
        at27 = float(np.interp(27.0, xs, [curve[x] for x in xs]))
        print(f"\n== 전환 난이도 분해 (B = 태스크 7 인 쌍들)")
        print(f"깨끗한 장면에서 27cm 떨어졌을 때 (보간): **{100 * at27:.0f}%**")
        print(f"{'쌍':>6s}{'실제 전환':>10s}{'예측(27cm)':>12s}{'차이':>8s}")
        for pair, rate in [("8→7", 0.55), ("4→7", 0.50), ("9→7", 0.00)]:
            print(f"{pair:>6s}{100 * rate:9.0f}%{100 * at27:11.0f}%{100 * (rate - at27):+7.0f}%p")
        easy = [0.55, 0.50]
        gap = float(np.mean(easy)) - at27
        if gap > 0.15:
            print(f"→ 쉬운 쌍은 예측보다 **{100 * gap:.0f}%p 높다**. 전환 시점의 팔은 "
                  "태스크를 수행하다 도달한 **자연스러운 자세**라, 임의로 옮겨 놓은 자세보다 쉽다")
        elif gap < -0.15:
            print(f"→ 쉬운 쌍이 예측보다 {100 * -gap:.0f}%p 낮다. 전환에는 위치 말고 다른 어려움이 있다")
        else:
            print("→ 쉬운 쌍은 예측과 거의 같다: 전환의 어려움은 팔 위치가 거의 전부")
        print("→ 9→7 만 0% 로 크게 벗어난다: 그 쌍에는 **위치 말고 다른 어려움**이 있다")

    # ---- 위치 vs 회전 비대칭 ----
    print("\n== 위치 vs 회전: 어느 쪽이 더 치명적인가")
    ycurve0 = {**KNOWN_YAW, **{r["yaw"]: r["rate"] for r in yaw_only}}
    pos_ok = [k for k, v in curve.items() if v >= 0.25]
    yaw_ok = [k for k, v in ycurve0.items() if v >= 0.25]
    if pos_ok and yaw_ok:
        print(f"성공률 25% 이상을 유지하는 한계:  위치 **{max(pos_ok):.0f}cm**  /  회전 **{max(yaw_ok):.0f}도**")
    yzero = sorted(k for k, v in ycurve0.items() if v < 0.05)
    if yzero:
        print(f"회전은 **{min(yzero):.0f}도부터 계속 0%** — 완만히 줄지 않고 절벽처럼 떨어진다")
    print("전환 시점의 실제 이탈: 위치 27cm(한계 근처) + 회전 8도(안전 구간)")
    print("→ **전환 자체는 회전이 문제가 아니다. 위치가 문제다**")
    print("→ 그런데 A 재개는 반대다: ret_pos(위치만 복귀) 7% vs retreat(위치+회전) 62%")
    print("   ⇒ 예측: **B 를 끝낸 시점의 손목 회전은 30도를 넘어 있다** (그래서 위치만 되돌려선 안 된다)")
    print("     검증: eef_rotvec 이 기록된 새 궤적에서 A2 시작 시점의 회전 이탈을 잰다")

    print("\n== 역커리큘럼에 쓸 수 있는 범위")
    ok = [r for r in off_only if r["rate"] >= 0.2]
    if ok:
        print(f"성공률 20% 이상인 최대 교란: **{max(r['off'] for r in ok):.0f}cm** "
              f"→ 여기까지는 자기 궤적으로 데이터를 모을 수 있다")
    zero = [r for r in off_only if r["rate"] < 0.05]
    if zero:
        print(f"사실상 0% 인 교란: {min(r['off'] for r in zero):.0f}cm 이상 → 단계적으로 넓혀야 한다")

    # ---- 그래프 ----
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150, facecolor=SURF)
    for ax in axs:
        ax.set_facecolor(SURF)
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(INK2); ax.tick_params(colors=INK2, length=0)
        ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    axs[0].plot(xs, [100 * curve[x] for x in xs], color="#2a78d6", lw=2.5, marker="o", ms=5)
    axs[0].axvline(27, color="#eb6834", ls="--", lw=1.5)
    axs[0].text(27.3, 90, "전환 시점 27cm", color="#eb6834", fontsize=9)
    axs[0].set_xlabel("초기 위치에서 옮긴 거리 (cm)", color=INK2)
    axs[0].set_ylabel("태스크 성공률 (%)", color=INK2); axs[0].set_ylim(0, 105)
    axs[0].set_title("위치 교란", color=INK, loc="left", fontsize=11)

    ycurve = {**KNOWN_YAW, **{r["yaw"]: r["rate"] for r in yaw_only}}
    yx = sorted(ycurve)
    axs[1].plot(yx, [100 * ycurve[x] for x in yx], color="#1baf7a", lw=2.5, marker="o", ms=5)
    axs[1].axvline(8, color="#eb6834", ls="--", lw=1.5)
    axs[1].text(8.5, 90, "전환 시점 8도", color="#eb6834", fontsize=9)
    axs[1].set_xlabel("손목 회전 (도)", color=INK2)
    axs[1].set_ylabel("태스크 성공률 (%)", color=INK2); axs[1].set_ylim(0, 105)
    axs[1].set_title("회전 교란", color=INK, loc="left", fontsize=11)

    fig.suptitle("어디까지 교란해도 정책이 작동하나 (태스크 7, 조건당 10회)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.text(0.01, 0.005, "전환 시점의 실제 이탈: 위치 27cm, 회전 8도. 데이터를 모을 수 있는 범위를 정하는 데 쓴다",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    Path("outputs/report").mkdir(parents=True, exist_ok=True)
    fig.savefig("outputs/report/boundary.png", facecolor=SURF)
    json.dump({"rows": rows, "off_curve": {str(k): v for k, v in curve.items()},
               "yaw_curve": {str(k): v for k, v in ycurve.items()}},
              open("outputs/report/boundary.json", "w"), ensure_ascii=False, indent=2)
    print("\n저장: outputs/report/boundary.png, boundary.json")


if __name__ == "__main__":
    main()
