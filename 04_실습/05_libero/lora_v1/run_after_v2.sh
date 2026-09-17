#!/usr/bin/env bash
# v2 파이프라인이 끝나면 이어서: 학습 없는 전환 전략 두 가지(rtc, bon)를 원본 모델로 평가
#  - rtc: Real-Time Chunking (NeurIPS 2025)
#  - bon: Best-of-N + 성공 궤적 지도 (Q-Planning 축소판)
# 비교 기준: 9/14 flush 결과 (같은 쌍·시점·에피소드)
set -u
cd "$(dirname "$0")"
log(){ echo "=== $(date +%H:%M:%S) $*"; }
log "v2 종료 대기"
until grep -q 'DONE' outputs/pipeline_v2.log 2>/dev/null; do sleep 60; done

log "성공 궤적 지도 갱신 (v2 데이터 포함)"
.venv/bin/python build_success_manifold.py > outputs/manifold_stats.json 2>&1

log "rtc / bon 평가 (원본 모델, 그 자리 전환)"
{
  for s in rtc bon; do
    for p in 8:7 4:7 1:5 2:5; do for k in grasp:3 grasp:20; do
      echo "--task-a ${p%:*} --task-b ${p#*:} --switch-at $k --strategy $s"
    done; done
  done
} | xargs -P 2 -I{} sh -c '.venv/bin/python switch_experiment.py {} --episodes 10 --out outputs/switch 2>&1 | grep -E "^SUMMARY|Traceback|Error" | grep -v EGL'
log "DONE"
