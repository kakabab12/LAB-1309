#!/usr/bin/env python
"""
조건별 맴돌기(policy idling) 비율 — 성공률과 다른 각도에서 방법을 본다

왜 이걸 보나
  전환 실패의 **80%가 맴돌기**다 (좁은 영역에서 왕복하며 진행 못 함).
  그래서 방법이 통한다면 **맴돌기가 먼저 줄어야** 한다.

  성공률은 0/1 이라 10회로는 53%p 미만을 못 본다.
  맴돌기 비율은 **에피소드마다 연속적으로** 재므로 같은 표본에서 더 민감하다.

    성공률 그대로 + 맴돌기 ↓  →  방향은 맞다. 표본만 부족한 것일 수 있다
    성공률 그대로 + 맴돌기 그대로 →  아무 일도 안 일어나고 있다
    성공률 그대로 + 맴돌기 ↑   →  해롭다

판정 기준 (2026-09-18 실패의 정체 보고서와 동일)
  B 구간에서 **2초(40스텝) 동안 이동 범위가 3cm 미만**인 구간이 있으면 맴돌았다고 본다.

쓰는 법
  python idling_by_cond.py [전환시점]      (기본: grasp3)
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

WIN = 40      # 2초 (20Hz)
RANGE_CM = 3  # 이 범위 안에서만 움직이면 맴돈 것


def idled(pos, phase):
    """B 구간에서 맴돈 적이 있나. (있나, 맴돈 스텝 비율) 반환."""
    b = np.nonzero(phase == "B")[0]
    if len(b) < WIN:
        return False, 0.0
    p = pos[b]
    hit = 0
    for t in range(len(p) - WIN):
        w = p[t:t + WIN]
        if 100 * float(np.max(np.ptp(w, axis=0))) < RANGE_CM:
            hit += 1
    return hit > 0, hit / max(len(p) - WIN, 1)


def label(name):
    """파일 이름에서 조건 이름을 뽑는다."""
    m = re.match(r"A\d+_B\d+_grasp\d+_flush(.*)_ep\d+$", name)
    if m is None:
        return None
    s = m.group(1)
    if s == "":
        return "기준(w=1)"
    m2 = re.match(r"^_cfg([0-9.]+)$", s)
    if m2:
        return f"증폭 w={float(m2.group(1)):g}"
    m3 = re.match(r"^_rep(\d+)$", s)
    if m3:
        return f"반복 k={m3.group(1)}"
    return None  # 다른 설정(지연·노이즈 등)은 섞지 않는다


def main():
    tim = sys.argv[1] if len(sys.argv) > 1 else "grasp3"
    rows = {}
    for d in ("outputs/switch", "outputs/cfg"):
        for f in glob.glob(f"{d}/traj/*_{tim}_*.npz"):
            lab = label(Path(f).stem)
            if lab is None:
                continue
            z = np.load(f, allow_pickle=True)
            if "phase" not in z:
                continue
            did, frac = idled(z["pos"], z["phase"])
            rows.setdefault(lab, []).append((did, frac))

    if not rows:
        print(f"'{tim}' 궤적이 없습니다.")
        return

    order = sorted(rows, key=lambda k: (k != "기준(w=1)", k))
    print(f"== B 구간 맴돌기 비율 — 전환 시점 {tim}")
    print(f"   (판정: {WIN / 20:.0f}초 동안 이동 범위 {RANGE_CM}cm 미만)\n")
    print(f"{'조건':<14}{'n':>5}{'맴돈 에피소드':>14}{'맴돈 시간 비중':>16}")
    base = None
    for k in order:
        v = rows[k]
        pct = 100 * float(np.mean([a for a, _ in v]))
        frac = 100 * float(np.mean([b for _, b in v]))
        if k == "기준(w=1)":
            base = pct
        print(f"{k:<14}{len(v):5d}{pct:13.0f}%{frac:15.0f}%")

    if base is None:
        print("\n기준선 궤적이 없어 비교할 수 없습니다.")
        return
    print("\n== 기준선 대비")
    for k in order:
        if k == "기준(w=1)":
            continue
        pct = 100 * float(np.mean([a for a, _ in rows[k]]))
        d = pct - base
        note = ("✅ 맴돌기가 줄었다 — 방향은 맞다" if d <= -10 else
                "❌ 맴돌기가 늘었다 — 해롭다" if d >= 10 else
                "– 거의 그대로 — 아무 일도 안 일어난다")
        print(f"  {k:<14}{d:+5.0f}%p   {note}")
    print("\n맴돌기는 에피소드마다 연속적으로 재므로, 성공률보다 적은 표본에서도 신호가 보인다.")
    print("성공률이 안 움직여도 맴돌기가 줄었다면 표본을 늘려 볼 가치가 있다.")


if __name__ == "__main__":
    main()
