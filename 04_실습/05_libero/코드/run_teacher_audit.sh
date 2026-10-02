#!/usr/bin/env bash
# 10/2 — 목표 95% 상향에 따라 선생(스크립트 전문가) 점검: 평가 장면 20~49, 노이즈 없음, 28개 항목
# v6c 학습이 끝난 뒤(메모리 확보) 시작. 프로세스 3개 (각 약 2.4GB).
cd "$(dirname "$0")"
PY=.venv/bin/python
until grep -q "\[2\]" outputs/v6/run_v6c.log; do sleep 60; done
mkdir -p outputs/teacher_audit
setsid $PY teacher_audit.py --tasks 0 1 2 3 4 5 6 7 8 9 --out outputs/teacher_audit/tasks.json > outputs/teacher_audit/tasks.log 2>&1 &
setsid $PY teacher_audit.py --pairs 8:0,8:3,8:5,8:7,8:9,4:5 --out outputs/teacher_audit/pairs1.json > outputs/teacher_audit/pairs1.log 2>&1 &
setsid $PY teacher_audit.py --pairs 4:9,1:7,8:1,8:4,1:8,4:1 --out outputs/teacher_audit/pairs2.json > outputs/teacher_audit/pairs2.log 2>&1 &
wait
echo "=== $(date +%H:%M) 선생 점검 끝"
