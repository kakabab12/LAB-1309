#!/usr/bin/env python
"""
집기 정밀도를 직접 잰다 — 학습이 **무엇을** 고쳤는지 보려면 성공률만으로는 부족하다

왜 필요한가 (2026-09-23)
  전환 뒤 팔은 목표 물체 5cm 까지 가고 집는 동작도 내는데 못 집는다.
  그리퍼가 **닫히는 순간**을 재 보니:

      교란 없이 성공한 집기   위치 오차 0.0cm (기준),  손목 6.9도
      전환 뒤                위치 오차 +3.1cm,        손목 15.7도

  **가기는 간다. 마지막 정렬이 안 된다.**

  LoRA 4차가 성공률을 올린다면, **이 오차가 줄어서** 올라야 한다.
  오차가 그대로인데 성공률만 올랐다면 다른 이유(운·지름길)이고, 일반화를 기대할 수 없다.

무엇을 재나
  그리퍼가 열림→닫힘으로 바뀌는 시점마다
    ① 목표 물체까지의 거리   ② 손목 회전이 기준에서 벗어난 각도
  를 기록한다.

⚠️ 끝단 좌표의 기준점이 손가락 끝이 아니라 손목이라 **상수 오차**가 있다.
   그래서 **교란 없이 성공한 집기**를 '오차 0' 기준선으로 삼는다.

쓰는 법
  python grasp_precision.py outputs/regrasp [outputs/pose_v4s1 ...]
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

OPEN = 0.035


def closes(src):
    """{(task, offset): [회전도, ...]}

    그리퍼가 닫히는 순간의 손목 회전 이탈을 잰다.
    그리퍼 정보가 없는 옛 궤적(2026-09-23 수정 이전)은 **정책 구간 전체의 중앙값**으로 대신한다.
    "집는 순간"만큼 날카롭지는 않지만, 같은 방식끼리 비교하면 경향은 볼 수 있다.
    """
    out, no_grip = {}, 0
    for f in sorted(glob.glob(f"{src}/traj/T*.npz")):
        m = re.match(r"T(\d+)_off([0-9.]+)_yaw([0-9.]+)_ep(\d+)$", Path(f).stem)
        if not m:
            continue
        task, off = int(m.group(1)), float(m.group(2))
        z = np.load(f, allow_pickle=True)
        if "eef_rotvec" not in z:
            continue
        pos, rv = z["pos"], z["eef_rotvec"]
        has_grip = "gripper_qpos" in z
        g = z["gripper_qpos"] if has_grip else np.zeros(len(pos))
        if len(pos) < 5 or len(rv) != len(pos):
            continue
        if not has_grip:
            no_grip += 1
        # 스크립트로 자세를 만드는 구간(R)은 빼고 정책 구간(A)만 본다
        ph = z["phase"] if "phase" in z else np.array(["A"] * len(g))
        idx = np.nonzero(ph == "A")[0]
        if len(idx) < 5:
            continue
        g, pos, rv = g[idx], pos[idx], rv[idx]
        h = Rotation.from_rotvec(z["home_rotvec"])
        rot = np.degrees(np.linalg.norm((Rotation.from_rotvec(rv) * h.inv()).as_rotvec(), axis=-1))
        if has_grip:
            for c in np.nonzero((g[:-1] >= OPEN) & (g[1:] < OPEN))[0]:
                out.setdefault((task, off), []).append(float(rot[c + 1]))
        else:
            out.setdefault((task, off), []).append(float(np.median(rot)))
    if no_grip:
        print(f"  ⚠️ {src}: 궤적 {no_grip}개에 그리퍼 정보가 없어 "
              f"**정책 구간 전체 중앙값**으로 대신했습니다 (2026-09-23 수정 이전 실행분)")
    return out


def main():
    srcs = sys.argv[1:] or ["outputs/regrasp"]
    print("== 그리퍼가 닫히는 순간의 손목 회전 이탈 (정렬 오차)\n")
    print("   교란이 커질수록 커지면 = 정렬이 무너지는 것")
    print("   학습 뒤 작아지면 = **학습이 정렬을 고쳤다**는 직접 증거\n")

    for src in srcs:
        d = closes(src)
        if not d:
            print(f"  {src}: 궤적이 없습니다 "
                  f"(--save-traj 로 돌렸는지, gripper_qpos 가 저장됐는지 확인)")
            continue
        print(f"  [{src}]")
        offs = sorted({o for _, o in d})
        tasks = sorted({t for t, _ in d})
        print(f"  {'태스크':<8}" + "".join(f"{str(int(o)) + 'cm':>10}" for o in offs))
        for t in tasks:
            line = f"  T{t:<7}"
            for o in offs:
                v = d.get((t, o))
                line += f"{np.median(v):9.1f}도" if v else f"{'-':>10}"
            print(line)
        print()

    print("읽는 법")
    print("  ⚠️ **태스크끼리 비교하면 안 된다.** 태스크마다 잡는 자세가 달라 기준값이 다르다")
    print("     (예: T9 와인병은 손목을 크게 돌려 잡으므로 원래 값이 크다)")
    print("  → **같은 태스크의 교란별 변화**, 그리고 **학습 전후 같은 칸**만 비교한다\n")
    print("  · 교란 0cm 의 값이 그 태스크의 **정상 집기 기준선**")
    print("  · 교란이 커질수록 이 값이 커지면, 성공률 하락의 원인이 정렬임을 뒷받침한다")
    print("  · 학습 전후를 나란히 놓고 **같은 교란에서 값이 줄었는지** 본다")
    print("\n⚠️ 목표 물체까지의 거리는 pose_sensitivity 궤적에 물체 위치가 없어 못 잰다.")
    print("   전환 실험(outputs/switch)에는 objects_at_switch 가 있어 거리까지 잴 수 있다.")


if __name__ == "__main__":
    main()
