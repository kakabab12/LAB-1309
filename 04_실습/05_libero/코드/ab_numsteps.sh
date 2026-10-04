#!/usr/bin/env bash
# 연구실 GPU(1080 Ti)에서 지연 줄이기·지연 손해 확인 (10/4, 줄인 판): PiPER 1차 모델이 어느 정도 해내는 항목만
#   기준(반복 10, 지연 11)은 1차 새 배치 평가(장면 1000~1019) 결과를 그대로 쓴다.
#   "10 0": 로봇이 기다려 줄 때 / "5 8": 반복 5단계 (약 0.40초 = 8스텝)
cd "$(dirname "$0")"
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
POL=outputs/piper_s1_model/merged
echo "=== $(date +%m/%d\ %H:%M) 비교 시작"
for cfg in "10 0" "5 8"; do
  set -- $cfg; NS=$1; LAT=$2; OUT=outputs/piper_s1_ns${NS}_lat$LAT
  EV="--episodes 10 --start-episode 1000 --latency-steps $LAT --n-action-steps 1 --num-steps $NS"
  [ "$LAT" -gt 0 ] && EV="$EV --ttrtc"
  for t in 1 4 7; do $PY piper_sim/eval_piper.py --policy $POL --task-a $t --strategy none $EV --out ${OUT}_forget 2>&1 | grep -E "$F"; done
  $PY piper_sim/eval_piper.py --policy $POL --task-a 8 --task-b 1 --switch-at grasp:3 --strategy keep $EV --out ${OUT}_switch 2>&1 | grep -E "$F"
done
echo "=== $(date +%m/%d\ %H:%M) 끝"
