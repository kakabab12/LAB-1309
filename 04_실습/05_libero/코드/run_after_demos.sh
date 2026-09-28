#!/usr/bin/env bash
# 녹화(demos.pkl)가 끝나면: 물체 기준 접근 + 정책  vs  재생 전문가 끝까지 — B5·B9·B3
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
echo "=== $(date +%H:%M:%S) 시작 (시작점: 접촉점 15cm 안 + 초기 자세 10cm 밖)"
$PY test_putdown.py --pairs 8:5,8:9,8:3 --episodes 10 --approach --approach-src demo \
  --out outputs/expert/putdown_demoapproach.json 2>&1 | grep -E "  ==|Traceback|Error" &
$PY test_replay.py --pairs 8:5,8:9,8:3 --episodes 10 --out outputs/expert/replay.json 2>&1 | grep -E "  ==|Traceback|Error" &
wait
echo "=== $(date +%H:%M:%S) 끝"
