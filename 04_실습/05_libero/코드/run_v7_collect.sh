#!/usr/bin/env bash
# 10/2 밤 — 고친 시범 프로그램(접시 바깥 밀기, 서랍 전 그릇 옮기기, 처음 자세 피해 가기)으로
# 학습용 새 무작위 배치(장면 2000번대)에서 시범을 모은다. 평가 전용 1000번대는 쓰지 않는다.
# v6c 학습이 끝나 메모리가 비면 시작, 수집기 2개 (mem_guard.sh 가 메모리 5GB 아래면 가장 새 수집기를 멈춘다).
cd "$(dirname "$0")"
PY=.venv/bin/python
while ps -eo comm,args | awk '$1=="python" && /train_lora\.py|teacher_audit\.py/' | grep -q .; do sleep 60; done
echo "=== $(date +%m/%d\ %H:%M) 정상 수행 시범 (10개 태스크 × 장면 2000~2059)"
setsid $PY collect_v6.py normal --tasks 0 1 2 3 4 --episodes 2000-2059 --dart 0.1 --seed 21 --out data/v7_normal > outputs/v6/v7_normal_a.log 2>&1 &
setsid $PY collect_v6.py normal --tasks 5 6 7 8 9 --episodes 2000-2059 --dart 0.1 --seed 22 --out data/v7_normal > outputs/v6/v7_normal_b.log 2>&1 &
wait
echo "=== $(date +%m/%d\ %H:%M) 전환·재개 시범 (12쌍 × 장면 2000~2039)"
setsid $PY collect_v6.py switch --pairs 8:0,8:3,8:5,8:7,8:9,4:5 --episodes 2000-2039 --dart 0.1 --seed 23 --out data/v7_switch > outputs/v6/v7_switch_a.log 2>&1 &
setsid $PY collect_v6.py switch --pairs 4:9,1:7,8:1,8:4,1:8,4:1 --episodes 2000-2039 --dart 0.1 --seed 24 --out data/v7_switch > outputs/v6/v7_switch_b.log 2>&1 &
wait
echo "=== $(date +%m/%d\ %H:%M) 수집 끝: 정상 $(ls data/v7_normal/episodes | wc -l), 전환 $(ls data/v7_switch/episodes | grep -c npz)"
