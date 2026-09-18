#!/usr/bin/env python
"""
예측 검증 — B 를 끝낸 시점의 손목 회전은 30도를 넘어 있는가

왜 재나 (2026-09-18)
  깨끗한 장면에서 손목만 돌려 놓고 시작하면
      20도 → 60%,  **30도 → 0%,  40도 → 0%,  50도 → 0%**
  회전은 위치와 달리 **절벽**이다 (위치는 25cm 에서도 29%).

  그런데 A 재개는 모든 전략에서 **0%** 이고,
      ret_pos (위치만 1.1cm 까지 복귀) → 7%
      retreat (위치+회전 복귀)         → 62%
  위치를 거의 완벽히 되돌려도 안 되고 회전까지 되돌려야 된다.

예측
  **A2 시작 시점(= B 종료 시점)의 손목 회전 이탈 > 30도**
  맞으면: A 재개 0% 는 "회전이 절벽 너머에 있어서" 로 설명된다
  틀리면(20도 이내): 이 설명은 폐기하고 다른 이유를 찾아야 한다

쓰는 법
  python rotation_at_resume.py                 # outputs/resume600
  python rotation_at_resume.py outputs/switch  # 다른 폴더
필요 조건
  궤적 npz 에 eef_rotvec 이 있어야 한다 (2026-09-18 16:04 이후 실행분)
"""
import glob
import sys
from pathlib import Path

import numpy as np

CLIFF = 30.0  # 이 각도부터 깨끗한 장면에서도 0%
SAFE = 20.0   # 이 각도까지는 60% 이상


def rot_deg(a, b):
    """두 회전벡터 사이의 각도 (도). 상대 회전의 크기."""
    from scipy.spatial.transform import Rotation
    rel = Rotation.from_rotvec(b) * Rotation.from_rotvec(a).inv()
    return float(np.degrees(np.linalg.norm(rel.as_rotvec())))


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "outputs/resume600"
    files = sorted(glob.glob(f"{src}/traj/*.npz"))
    if not files:
        print(f"{src}/traj 에 궤적이 없습니다."); return

    rows, skipped = [], 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "eef_rotvec" not in d:
            skipped += 1
            continue
        ph, rv, pos = d["phase"], d["eef_rotvec"], d["pos"]
        home_rv, home_pos = d["home_rotvec"], d["home_pos"]
        idx = {k: np.nonzero(ph == k)[0] for k in ("B", "A2")}
        if not len(idx["A2"]):
            continue
        t = int(idx["A2"][0])  # A 재개가 시작되는 첫 스텝
        rows.append({
            "file": Path(f).name,
            "rot_deg": rot_deg(home_rv, rv[t]),
            "pos_cm": 100 * float(np.linalg.norm(pos[t] - home_pos)),
            # 비교용: 전환 시점 (B 의 첫 스텝)
            "rot_switch": rot_deg(home_rv, rv[int(idx["B"][0])]) if len(idx["B"]) else np.nan,
            "pos_switch": (100 * float(np.linalg.norm(pos[int(idx["B"][0])] - home_pos))
                           if len(idx["B"]) else np.nan),
        })

    if skipped:
        print(f"⚠️ eef_rotvec 이 없는 오래된 궤적 {skipped}개는 건너뜀 "
              f"(2026-09-18 16:04 이전 실행분)")
    if not rows:
        print("A2(재개) 구간이 있는 궤적이 없습니다. --max-steps 를 늘려 재개까지 돌려야 합니다.")
        return

    rot = np.array([r["rot_deg"] for r in rows])
    pos = np.array([r["pos_cm"] for r in rows])
    sw_rot = np.array([r["rot_switch"] for r in rows])
    sw_pos = np.array([r["pos_switch"] for r in rows])

    print(f"== 궤적 {len(rows)}개 ({src})\n")
    print(f"{'시점':>18s}{'회전 이탈':>12s}{'위치 이탈':>12s}")
    print(f"{'전환 (B 시작)':>18s}{np.nanmedian(sw_rot):9.1f}도{np.nanmedian(sw_pos):9.1f}cm")
    print(f"{'재개 (A2 시작)':>18s}{np.median(rot):9.1f}도{np.median(pos):9.1f}cm")
    print(f"\n재개 시점 회전 분포: 중앙값 {np.median(rot):.1f}도, "
          f"25~75% {np.percentile(rot, 25):.1f}~{np.percentile(rot, 75):.1f}도, "
          f"최소 {rot.min():.1f}도, 최대 {rot.max():.1f}도")
    over = 100 * float(np.mean(rot > CLIFF))
    under = 100 * float(np.mean(rot <= SAFE))
    print(f"{CLIFF:.0f}도(절벽)를 넘은 비율: **{over:.0f}%**   "
          f"{SAFE:.0f}도(안전) 이내 비율: {under:.0f}%")

    print("\n== 판정")
    if np.median(rot) > CLIFF:
        print(f"✅ **예측이 맞다.** 재개 시점 회전이 중앙값 {np.median(rot):.0f}도로 절벽({CLIFF:.0f}도)을 넘는다.")
        print("   → A 재개 0% 는 '회전이 정책이 견딜 수 있는 범위 밖' 으로 설명된다.")
        print("   → 위치만 되돌리는 대책(ret_pos 7%)이 실패한 이유이기도 하다.")
    elif np.median(rot) <= SAFE:
        print(f"❌ **예측이 틀렸다.** 재개 시점 회전이 중앙값 {np.median(rot):.0f}도로 안전 구간({SAFE:.0f}도) 안이다.")
        print("   → A 재개 0% 를 회전으로 설명할 수 없다. 다른 이유를 찾아야 한다.")
        print(f"   → 남은 후보: 위치({np.median(pos):.0f}cm, 경계 25cm 를 크게 넘음), 장면 변화(B 가 물체를 옮김)")
    else:
        print(f"⚠️ **애매하다.** 중앙값 {np.median(rot):.0f}도로 안전({SAFE:.0f})과 절벽({CLIFF:.0f}) 사이다.")
        print("   → 회전만으로는 0% 를 설명하기 어렵다. 위치와 함께 봐야 한다.")

    print(f"\n참고: 재개 시점 위치 이탈 중앙값 {np.median(pos):.0f}cm — "
          f"깨끗한 장면 경계(25cm, 29%)를 {'넘는다' if np.median(pos) > 25 else '넘지 않는다'}")
    print("      위치와 회전 둘 다 범위 밖이면 어느 하나만 고쳐서는 안 된다는 뜻이다")


if __name__ == "__main__":
    main()
