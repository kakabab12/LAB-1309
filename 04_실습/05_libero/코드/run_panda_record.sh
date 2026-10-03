#!/usr/bin/env bash
# Panda 기록용 성적표 (10/3): v6c3 를 실제 배치 조건(지연 11 + 앞부분 제공, 계속 계산)으로 새 배치 1000~1019 에서.
# PiPER 학습과 같이 돌므로 프로세스 2개만 (메모리: 학습 13GB + 평가 2×3GB)
cd "$(dirname "$0")"
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
while pgrep -f "audit_piper[.]py" >/dev/null; do sleep 60; done
POL=outputs/v6c3_model/merged; OUT=panda_v6c3_rec
EV="--episodes 20 --start-episode 1000 --latency-steps 11 --ttrtc --n-action-steps 1"
log "Panda v6c3 기록용 평가 시작"
( for t in 0 1 2 3 4 5 6 7 8 9; do $PY switch_experiment.py --policy $POL --task-a $t --strategy none $EV --out outputs/${OUT}_forget 2>&1 | grep -E "$F"; done ) &
( for p in 8:0 8:3 8:5 8:7 8:9 4:5 4:9 1:7 8:1 8:4 1:8 4:1; do $PY switch_experiment.py --policy $POL --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy keep $EV --out outputs/${OUT}_switch 2>&1 | grep -E "$F"; done ) &
wait
$PY scoreboard.py --forget outputs/${OUT}_forget --switch outputs/${OUT}_switch --name "Panda v6c3 기록 (새 배치 20장면, 지연 11)" \
  --png outputs/media/score_${OUT}.png --md outputs/v6/score_${OUT}.md | tail -1
log "끝"
