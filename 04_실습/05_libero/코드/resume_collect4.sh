#!/usr/bin/env bash
# 메모리 때문에 멈춘 4차 수집을, 반복 단계 시험(평가 2개)이 끝나면 다시 시작 (이미 모은 장면은 건너뜀)
cd "$(dirname "$0")"
while ps -eo args | grep -qE "^bash \./run_ns_test"; do sleep 60; done
echo "=== $(date +%m/%d\ %H:%M) 4차 수집 다시 시작" >> outputs/piper/blk_collect4.log
exec ./run_blk_collect4.sh >> outputs/piper/blk_collect4.log 2>&1
