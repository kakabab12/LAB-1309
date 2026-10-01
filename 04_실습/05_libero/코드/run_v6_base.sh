#!/usr/bin/env bash
# 6차 평가가 끝나면 원래 모델도 장면 30~49 에서 평가 (20~29 는 v5eval_*_base 에 있다) → 같은 장면 30회로 나란히 비교
cd "$(dirname "$0")"
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
while ps -eo comm,args | awk '$1=="bash" && /run_v6\.sh/' | grep -q .; do sleep 120; done
EV="--episodes 20 --start-episode 30"
POL=HuggingFaceVLA/smolvla_libero
( for t in 0 1 2 3 4 5 6 7 8 9; do $PY switch_experiment.py --policy $POL --task-a $t --strategy none $EV --out outputs/v6eval_forget_base 2>&1 | grep -E "$F"; done ) &
( for p in 8:0 8:3 8:5 8:7 8:9 4:5 4:9 1:7 8:1 8:4 1:8 4:1; do $PY switch_experiment.py --policy $POL --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy flush $EV --out outputs/v6eval_switch_base 2>&1 | grep -E "$F"; done ) &
wait
echo "=== $(date +%H:%M) 원래 모델 평가 끝"
