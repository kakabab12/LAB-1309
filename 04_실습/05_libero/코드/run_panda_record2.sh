#!/usr/bin/env bash
# Panda v6c3 기록용 성적표 (10/3, 이어서): PiPER 학습이 도는 동안에만 평가한다 (PiPER 평가·DAgger 와 GPU 메모리가 겹치지 않게).
#   이미 끝난 항목(JSON 있음)은 건너뛴다. 프로세스 2개.
cd "$(dirname "$0")"
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
POL=outputs/v6c3_model/merged; OUT=panda_v6c3_rec
EV="--episodes 20 --start-episode 1000 --latency-steps 11 --ttrtc --n-action-steps 1"
gate(){ until pgrep -f "train_lora[.]py.*piper_" >/dev/null; do sleep 120; done; }
run_t(){ for t in "$@"; do ls outputs/${OUT}_forget/A${t}_Bnone_* >/dev/null 2>&1 && continue; gate
  $PY switch_experiment.py --policy $POL --task-a $t --strategy none $EV --out outputs/${OUT}_forget 2>&1 | grep -E "$F"; done; }
run_p(){ for p in "$@"; do ls outputs/${OUT}_switch/A${p%:*}_B${p#*:}_* >/dev/null 2>&1 && continue; gate
  $PY switch_experiment.py --policy $POL --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy keep $EV --out outputs/${OUT}_switch 2>&1 | grep -E "$F"; done; }
log "Panda v6c3 기록용 평가 (PiPER 학습 중에만)"
( run_t 0 1 2 3 4 5 6 7 8 9 ) &
( run_p 8:0 8:3 8:5 8:7 8:9 4:5 4:9 1:7 8:1 8:4 1:8 4:1 ) &
wait
$PY scoreboard.py --forget outputs/${OUT}_forget --switch outputs/${OUT}_switch --name "Panda v6c3 기록 (새 배치 20장면, 지연 11)" \
  --png outputs/media/score_${OUT}.png --md outputs/v6/score_${OUT}.md | tail -1
log "끝"
