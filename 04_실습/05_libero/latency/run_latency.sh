#!/usr/bin/env bash
# 추론 지연 실험 (연구주제.md 2단계 "저사양"): RTC 의 원래 목적(지연 대응)에 맞는 공정한 비교
set -u
cd "$(dirname "$0")"
log(){ echo "=== $(date +%H:%M:%S) $*"; }
log "A. 전환 없음: none vs rtc_all × 지연 5·10·20 스텝"
{
  for L in 5 10 20; do for s in none rtc_all; do for t in 1 5 7 8; do
    echo "--task-a $t --strategy $s --switch-at step:9999 --latency-steps $L"
  done; done; done
  for s in flush keep rtc; do for p in 8:7 4:7 1:5; do
    echo "--task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy $s --latency-steps 10"
  done; done
} | xargs -P 2 -I{} sh -c '.venv/bin/python switch_experiment.py {} --episodes 10 --out outputs/latency 2>&1 | grep -E "^SUMMARY|Traceback" | grep -v EGL'
log "DONE"
