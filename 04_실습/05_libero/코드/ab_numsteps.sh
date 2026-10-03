#!/usr/bin/env bash
# 연구실 GPU(1080 Ti)에서 지연 줄이기 (10/3): 동작 만들기 반복 단계 10 → 5 → 3.
#   bench_split 비율로 환산한 지연: 10단계 0.56초(11스텝), 5단계 약 0.40초(8스텝), 3단계 약 0.35초(7스텝)
#   PiPER 1차 모델로 같은 장면(1000~1009)·같은 항목에서 비교. GPU 작업 4개를 넘지 않게 프로세스 1개.
cd "$(dirname "$0")"
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
POL=outputs/piper_s1_model/merged
until grep -q "\[1\]" outputs/piper/round_s2.log 2>/dev/null; do sleep 120; done    # 1차 평가가 끝난 뒤 (라운드 시작)
echo "=== $(date +%m/%d\ %H:%M) 반복 단계 비교 시작"
# "10 0": 지연 없음 — 지연이 얼마나 손해인지 (VLASH 같은 지연 대응을 더 넣을지 판단용)
for cfg in "10 11" "5 8" "3 7" "10 0"; do
  set -- $cfg; NS=$1; LAT=$2; OUT=outputs/piper_s1_ns${NS}_lat$LAT
  EV="--episodes 10 --start-episode 1000 --latency-steps $LAT --n-action-steps 1 --num-steps $NS"
  [ "$LAT" -gt 0 ] && EV="$EV --ttrtc"
  for t in 0 3 8; do $PY piper_sim/eval_piper.py --policy $POL --task-a $t --strategy none $EV --out ${OUT}_forget 2>&1 | grep -E "$F"; done
  for p in 8:3 8:7 8:5; do $PY piper_sim/eval_piper.py --policy $POL --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy keep $EV --out ${OUT}_switch 2>&1 | grep -E "$F"; done
done
echo "=== $(date +%m/%d\ %H:%M) 끝"
