#!/usr/bin/env bash
# A4/A1 접근 시험이 끝나면: 그릇 태스크(T1·T4·T8) 성공 궤적 녹화 → A 재개 시험
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
while pgrep -f "putdown_approach_a41" >/dev/null; do sleep 20; done
echo "=== $(date +%H:%M:%S) 그릇 태스크 녹화"
$PY record_demos.py --tasks 8 4 1 --episodes 10 --append 2>&1 | grep -E "  ==|저장:|Traceback|Error"
echo "=== $(date +%H:%M:%S) A 재개 시험"
$PY test_resume.py --episodes 10 2>&1 | grep -E "  ==|Traceback|Error"
echo "=== $(date +%H:%M:%S) 끝"
