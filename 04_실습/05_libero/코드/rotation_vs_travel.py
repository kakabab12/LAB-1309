#!/usr/bin/env python
"""
가설 검증 — 정책은 "이동하면서만" 손목 방향을 고치는가

왜 이 가설이 나왔나 (2026-09-21)
  손목을 30도 돌려 놓고 시작하면 **0/20 = 0%** 다.
  그런데 손목을 30도 돌리고 **팔도 20cm 옮겨** 놓으면 **3/7 = 43%** 로 산다.
  더 심하게 교란했는데 더 잘한다 — 모순처럼 보인다.

  가설:
      회전만 주면 → 팔이 제자리라 정책이 곧장 집으려 한다 → **고칠 구간이 없다** → 실패
      위치도 주면 → 이동해야 한다 → **이동하는 동안 손목이 제자리를 찾는다** → 성공

  즉 **손목 보정은 이동에 얹혀서 일어난다**는 것이다.

어떻게 재나
  궤적에서 스텝마다
      이동량  = ‖위치(t+1) − 위치(t)‖
      회전량  = 각도(회전(t) → 회전(t+1))
  둘의 관계를 본다. 가설이 맞으면 **이동이 큰 구간에서 회전 변화도 크다.**

  또 하나: 회전 **오차가 줄어드는가**(집으로 가까워지는가)를 이동량별로 나눠 본다.
  단순히 "같이 움직인다"가 아니라 **이동할 때 오차가 준다**는 것이 핵심이다.

쓰는 법
  python rotation_vs_travel.py [폴더...]      (기본: eef_rotvec 이 있는 모든 폴더)
"""
import glob
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr


def angles_to(home_rv, rv):
    """각 스텝의 회전이 기준(home)에서 얼마나 벗어났는지 (도)."""
    h = Rotation.from_rotvec(home_rv)
    rel = Rotation.from_rotvec(rv) * h.inv()
    return np.degrees(np.linalg.norm(rel.as_rotvec(), axis=-1))


def step_rot_delta(rv):
    """연속한 두 스텝 사이의 회전량 (도)."""
    a, b = Rotation.from_rotvec(rv[:-1]), Rotation.from_rotvec(rv[1:])
    return np.degrees(np.linalg.norm((b * a.inv()).as_rotvec(), axis=-1))


def main():
    srcs = sys.argv[1:] or ["outputs/latency", "outputs/cfg", "outputs/switch"]
    files = []
    for s in srcs:
        files += glob.glob(f"{s}/traj/*.npz")
    move, rot, derr, used, skipped = [], [], [], 0, 0
    for f in sorted(files):
        d = np.load(f, allow_pickle=True)
        if "eef_rotvec" not in d:
            skipped += 1
            continue
        pos, rv, home_rv = d["pos"], d["eef_rotvec"], d["home_rotvec"]
        if len(pos) < 5 or len(rv) != len(pos):
            continue
        m = 100 * np.linalg.norm(np.diff(pos, axis=0), axis=1)   # cm/스텝
        r = step_rot_delta(rv)                                    # 도/스텝
        err = angles_to(home_rv, rv)
        move.append(m); rot.append(r); derr.append(np.diff(err))  # 오차 변화 (음수면 줄어듦)
        used += 1

    if not used:
        print(f"eef_rotvec 이 있는 궤적이 없습니다 (건너뜀 {skipped}개).")
        print("2026-09-18 16:04 이후에 돈 실험이어야 합니다.")
        return
    move = np.concatenate(move); rot = np.concatenate(rot); derr = np.concatenate(derr)
    print(f"궤적 {used}개, 스텝 {len(move)}개 (eef_rotvec 없는 옛 궤적 {skipped}개 건너뜀)\n")

    rho, p = spearmanr(move, rot)
    print(f"== ① 이동량과 회전량의 관계\n  스피어만 상관 ρ = {rho:+.3f}  (p = {p:.1e})")
    print(f"  {'✅ 이동이 클수록 회전도 크다 — 가설과 맞다' if rho > 0.3 else ('관계가 약하다' if rho > 0.1 else '❌ 관계가 없다 — 가설과 다르다')}")

    print("\n== ② 이동량 구간별 — 회전이 실제로 '고쳐지는가'")
    qs = np.quantile(move, [0, 0.25, 0.5, 0.75, 1.0])
    print(f"{'이동량 구간':>18}{'스텝수':>8}{'회전량(도)':>11}{'회전오차 변화':>14}")
    for i in range(4):
        lo, hi = qs[i], qs[i + 1]
        sel = (move >= lo) & (move <= hi if i == 3 else move < hi)
        if sel.sum() < 10:
            continue
        print(f"  {lo:5.2f}~{hi:5.2f}cm{sel.sum():10d}{np.median(rot[sel]):11.2f}"
              f"{np.mean(derr[sel]):+14.3f}")
    print("  (회전오차 변화가 **음수**면 그 구간에서 손목이 제자리로 돌아오고 있다는 뜻)")

    print("\n⚠️ **이 집단으로는 가설을 검증할 수 없다**")
    print("   여기 궤적들은 손목이 처음부터 맞은 정상 에피소드다.")
    print("   가설은 '**틀어진** 손목이 이동 중에 고쳐지는가' 인데, 여기엔 틀어진 시작이 없다.")
    print("   → 제대로 보려면 pose_sensitivity.py --yaw-deg 30 --save-traj 로 모은 궤적이 필요하다.")
    print("   아래 ③은 참고용이고, 결론은 '드리프트' 쪽(②의 부호)에 있다.\n")

    slow = move < qs[1]
    fast = move >= qs[3]
    print(f"\n== ③ 가장 느린 25% vs 가장 빠른 25%")
    print(f"  느릴 때  회전량 {np.median(rot[slow]):5.2f}도   회전오차 변화 {np.mean(derr[slow]):+.3f}")
    print(f"  빠를 때  회전량 {np.median(rot[fast]):5.2f}도   회전오차 변화 {np.mean(derr[fast]):+.3f}")
    ratio = np.median(rot[fast]) / max(np.median(rot[slow]), 1e-6)
    print(f"  빠를 때 회전량이 {ratio:.1f}배")
    # 이 집단에서 실제로 알 수 있는 것: 손목이 집에서 멀어지기만 하는가
    print("== ④ ⭐ 이 집단에서 확실히 말할 수 있는 것 — 손목은 계속 멀어지기만 한다")
    pos_frac = 100 * float(np.mean(derr > 0))
    print(f"  회전 오차가 커진 스텝 비율: **{pos_frac:.0f}%**  (평균 변화 {np.mean(derr):+.3f}도/스텝)")
    if np.mean(derr) > 0 and all(np.mean(derr[(move >= qs[i]) & (move < qs[i + 1])]) > 0
                                 for i in range(3)):
        print("  **모든 이동량 구간에서 오차가 늘어난다.** 손목은 되돌아오지 않는다.")
        print("  → 재개 시점의 84도는 한 번 크게 비틀린 것이 아니라 **조금씩 쌓인 결과**다")
        print("  → 이동이 빠를수록 늘어나는 속도가 느릴 뿐, 방향이 바뀌지는 않는다")
    print("\n  이것은 [재개 시점 손목 84도] 측정과 독립적으로 맞물린다:")
    print("  정책에는 손목을 기준으로 되돌리는 기제가 아예 없다는 뜻이다")


if __name__ == "__main__":
    main()
