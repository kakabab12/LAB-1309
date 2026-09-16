#!/usr/bin/env bash
# 자세 민감도: 태스크 3개 × (위치 오프셋 6단계 + 손목 회전 3단계) × 10 에피소드
set -u
cd "$(dirname "$0")"
J=outputs/pose_jobs.txt; : > $J
for t in 7 5 1; do
  for o in 0 2 4 6 8 10; do echo "--task $t --offset-cm $o" >> $J; done
  for y in 10 20 30; do echo "--task $t --yaw-deg $y" >> $J; done
done
cat $J | xargs -P 3 -I{} sh -c '.venv/bin/python pose_sensitivity.py {} --episodes 10 --out outputs/pose 2>&1 | grep -E "^SUMMARY|Traceback|Error"'
echo DONE
