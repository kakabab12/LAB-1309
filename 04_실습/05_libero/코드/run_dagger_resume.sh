#!/usr/bin/env bash
# 10/2 — 재개 구간 DAgger: 전환 쌍에서 B 를 마친 뒤 원래 일로 돌아가는 동안 v6b 가 틀어지면 선생이 바로잡는다.
# 지금 도는 DAgger(p3·p4)가 끝나면 시작한다. 프로세스 2개.
cd "$(dirname "$0")"
PY=.venv/bin/python
while ps -eo comm,args | awk '$1=="python" && /dagger_v6\.py/' | grep -q .; do sleep 60; done
echo "=== $(date +%H:%M) 재개 DAgger 시작"
D="--policy outputs/v6b_model/merged --latency-steps 11 --episodes 0-19,50-69 --out data/v6_dagger_lat --resume"
setsid $PY dagger_v6.py $D --pairs 8:5,8:7,8:9 --seed 5 > outputs/v6/dagger/r1.log 2>&1 &
setsid $PY dagger_v6.py $D --pairs 4:5,4:9,1:7 --seed 6 > outputs/v6/dagger/r2.log 2>&1 &
wait
echo "=== $(date +%H:%M) 재개 DAgger 끝: $(ls data/v6_dagger_lat/episodes | grep -c '^DR')"
