#!/usr/bin/env bash
# 6차 중간 모델(4000스텝)로 미리 보기 — 몇 항목만, 한 번에 하나씩 (학습과 GPU 를 나눠 쓰므로)
cd "$(dirname "$0")"
PY=.venv/bin/python
POL=outputs/v6_model/step_4000
EV="--episodes 10 --start-episode 20"
for t in 0 8 5; do
  $PY switch_experiment.py --policy $POL --task-a $t --strategy none $EV --out outputs/v6early_forget 2>&1 | grep -E "SUMMARY|Traceback|Error"
done
for p in 8:7 8:3 8:0; do
  $PY switch_experiment.py --policy $POL --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy flush $EV --out outputs/v6early_switch 2>&1 | grep -E "SUMMARY|Traceback|Error"
done
echo "=== $(date +%H:%M) 미리 보기 끝"
