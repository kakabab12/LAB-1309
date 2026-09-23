#!/usr/bin/env python
"""
A 재개를 다시 센다 — '진짜 재개'와 '우연'을 나눈다

왜 (2026-09-23 발견)
  A1='그릇을 스토브 위에' + B7='스토브 켜기' 에서 A 재개가 **90%** 로 보였다.
  그런데 10회 중 **9회가 A2 를 0스텝 만에 '성공'** 했다.
  = B 를 하는 동안 스토브 근처에 그릇을 놓아, A 의 목표가 **저절로 달성**된 것이다.
  돌아가서 다시 해낸 것이 아니다.

  **A2 가 0스텝이면 '이미 이뤄져 있었다'는 뜻이므로 재개가 아니다.**

  이 구분을 안 하면 "장소가 겹치는 쌍"에서 성능이 부풀려진다.
  우리 연구의 두 번째 목표(중단한 일로 돌아오기)가 걸린 문제라 중요하다.
"""
import glob
import json
import re
from pathlib import Path


def main():
    tot = [0, 0, 0, 0]
    per = {}
    for f in sorted(glob.glob("outputs/switch/A*_B*_grasp3_flush.json")):
        m = re.match(r"A(\d+)_B(\d+)_grasp3_flush$", Path(f).stem)
        if not m:
            continue
        eps = [e for e in json.load(open(f))["episodes"] if e.get("switched")]
        if not eps:
            continue
        app = sum(bool(e.get("a_resume_success")) for e in eps)
        acc = sum(1 for e in eps
                  if e.get("a_resume_success") and (e.get("a_resume_steps") or 0) == 0)
        tot[0] += app; tot[1] += acc; tot[2] += app - acc; tot[3] += len(eps)
        if app:
            per[f"A{m.group(1)}→B{m.group(2)}"] = (app, acc, app - acc, len(eps))

    if not tot[3]:
        print("결과가 없습니다.")
        return
    n = tot[3]
    print("== A 재개 감사 (전환한 에피소드 전체)\n")
    print(f"  겉보기 성공        {tot[0]:3d}/{n}  = {100 * tot[0] / n:4.1f}%")
    print(f"  그중 우연 (0스텝)  {tot[1]:3d}      ← B 하는 동안 A 목표가 저절로 달성됨")
    print(f"  **진짜 재개**      {tot[2]:3d}/{n}  = {100 * tot[2] / n:4.1f}%")
    if per:
        print(f"\n{'쌍':<9}{'겉보기':>8}{'우연':>7}{'진짜':>7}")
        for k, (a, c, r, t) in sorted(per.items()):
            print(f"{k:<9}{a:5d}/{t:<3d}{c:6d}{r:6d}")
    print("\n⚠️ 겉보기 수치를 논문이나 보고서에 쓰면 안 된다.")
    print("   장소가 겹치는 쌍(A1→B7 등)에서 크게 부풀려진다.")


if __name__ == "__main__":
    main()
