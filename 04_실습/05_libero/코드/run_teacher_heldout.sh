#!/usr/bin/env bash
# 시범 프로그램 성공률 — 학생 최종 평가와 같은 새 배치(장면 1000~1099), 100장면씩, 노이즈 없음 (2026-10-02)
# v6c 학습이 끝나 메모리가 비면 시작. 프로세스 2개.
cd "$(dirname "$0")"
PY=.venv/bin/python
while ps -eo comm,args | awk '$1=="python" && /train_lora\.py/' | grep -q .; do sleep 60; done
while ps -eo comm,args | awk '$1=="python" && /teacher_audit\.py/' | grep -q .; do sleep 60; done
mkdir -p outputs/teacher_audit
setsid $PY teacher_audit.py --tasks 0 1 2 3 4 5 6 7 8 9 --pairs 8:0,8:3,8:5 --episodes 1000-1099 --out outputs/teacher_audit/heldout_a.json > outputs/teacher_audit/heldout_a.log 2>&1 &
setsid $PY teacher_audit.py --pairs 8:7,8:9,4:5,4:9,1:7,8:1,8:4,1:8,4:1 --episodes 1000-1099 --out outputs/teacher_audit/heldout_b.json > outputs/teacher_audit/heldout_b.log 2>&1 &
wait
echo "=== $(date +%H:%M) 시범 프로그램 새 배치 100장면 측정 끝"
