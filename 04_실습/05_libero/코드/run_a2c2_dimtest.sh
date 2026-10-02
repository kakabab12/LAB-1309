#!/usr/bin/env bash
# 10/2 가설 확인: A2C2 가 A8→B7(스토브 켜기)을 70→30% 로 떨어뜨린 것이 그리퍼 보정 때문인가
#   실패 장면에서 손은 손잡이까지 갔는데 그리퍼가 열린 채 끝났다 (traj 분석)
#   → 그리퍼 보정을 끄고(posrot) 같은 장면에서 다시 잰다. 단독 T8 이득이 남는지도 같이 본다.
# v6c 학습이 끝난 뒤(run_v6c.sh [2] 시작) 돈다.
cd "$(dirname "$0")"
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
until grep -q "\[2\]" outputs/v6/run_v6c.log; do sleep 60; done
EV="--episodes 10 --start-episode 20 --latency-steps 11 --a2c2 outputs/a2c2_v6b --a2c2-dims posrot"
$PY switch_experiment.py --policy outputs/v6b_model/merged --task-a 8 --task-b 7 --switch-at grasp:3 --strategy flush $EV --out outputs/v6ba_posrot 2>&1 | grep -E "$F"
$PY switch_experiment.py --policy outputs/v6b_model/merged --task-a 8 --strategy none $EV --out outputs/v6ba_posrot 2>&1 | grep -E "$F"
echo "=== $(date +%H:%M) 그리퍼 보정 끈 시험 끝"
