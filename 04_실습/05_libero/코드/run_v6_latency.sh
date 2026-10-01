#!/usr/bin/env bash
# 저사양 조건 평가 — 1080 Ti 실측 추론 지연(0.56초 = 11스텝)을 넣고 6차와 원래 모델을 비교 (연구주제.md 2단계)
# 원래 모델 평가(run_v6_base.sh)가 끝난 뒤에 돈다. 동시에 2개까지 (메모리).
cd "$(dirname "$0")"
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
while ps -eo comm,args | awk '$1=="bash" && /run_v6_base\.sh|run_v6\.sh/ && !/awk/' | grep -q .; do sleep 120; done
EV="--episodes 10 --start-episode 20 --latency-steps 11"
for POL in outputs/v6_model/merged HuggingFaceVLA/smolvla_libero; do
  TAG=$([ "$POL" = "HuggingFaceVLA/smolvla_libero" ] && echo base || echo v6)
  ( for t in 0 1 2 3 4 5 6 7 8 9; do $PY switch_experiment.py --policy $POL --task-a $t --strategy none $EV --out outputs/v6lat_forget_$TAG 2>&1 | grep -E "$F"; done ) &
  ( for p in 8:0 8:3 8:5 8:7 8:9 4:5 4:9 1:7 8:1 8:4 1:8 4:1; do $PY switch_experiment.py --policy $POL --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy flush $EV --out outputs/v6lat_switch_$TAG 2>&1 | grep -E "$F"; done ) &
  wait
done
echo "=== $(date +%H:%M) 지연 평가 끝"
