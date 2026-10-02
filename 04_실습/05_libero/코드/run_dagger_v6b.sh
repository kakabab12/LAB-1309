#!/usr/bin/env bash
# 10/2 — v6b 모델로 DAgger 수집 (실제 로봇 조건: 추론 지연 11스텝) + v6b 지연 평가
# GPU 4개 프로세스까지 (1080 Ti 11GB, 프로세스당 ~2.3GB). 학습과 겹치지 않는다.
cd "$(dirname "$0")"
PY=.venv/bin/python
POL=outputs/v6b_model/merged
D="--policy $POL --latency-steps 11 --episodes 0-19,50-69 --out data/v6_dagger_lat"
mkdir -p outputs/v6/dagger
setsid $PY dagger_v6.py $D --tasks 0 3 5 7 9 --seed 1 > outputs/v6/dagger/p1.log 2>&1 &
setsid $PY dagger_v6.py $D --tasks 1 4 6 8 2 --seed 2 > outputs/v6/dagger/p2.log 2>&1 &
setsid $PY dagger_v6.py $D --pairs 8:0,8:3,8:5,8:7,8:9,4:5,4:9,1:7,8:1,8:4,1:8,4:1 --seed 3 > outputs/v6/dagger/p3.log 2>&1 &
EV="--episodes 10 --start-episode 20 --latency-steps 11"
F='SUMMARY|Traceback|Error'
( for p in 8:0 8:3 8:5 8:7 8:9 4:5 4:9 1:7 8:1 8:4 1:8 4:1; do $PY switch_experiment.py --policy $POL --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy flush $EV --out outputs/v6blat_switch 2>&1 | grep -E "$F"; done
  for t in 0 1 2 3 4 5 6 7 8 9; do $PY switch_experiment.py --policy $POL --task-a $t --strategy none $EV --out outputs/v6blat_forget 2>&1 | grep -E "$F"; done ) > outputs/v6/v6blat.log 2>&1 &
wait
echo "=== $(date +%H:%M) DAgger·지연평가 끝"
