#!/usr/bin/env bash
# 그리퍼 유지(grip latch) 검증
#
# 발견 (2026-09-23)
#   "그릇을 서랍에 넣어라" 를 **그릇을 쥔 채** 받아도 정책은 10/10 그릇을 놓는다.
#   '지시가 바뀌면 일단 놓는다' 가 학습된 반사다.
#   B 의 목표가 물체인 쌍은 1.4%, 가구·기구인 쌍은 44.6% (30배).
#
# 그래서 전환 이후 N 스텝 동안 그리퍼만 닫힌 채로 유지해 본다.
# 팔은 정책이 내는 그대로라 제약(초기자세 복귀 금지·자연스러움)을 지킨다.
#
# 가장 깨끗한 경우는 **A2→B9**: 와인병을 쥔 채 "와인병을 선반에 놓아라".
# 한 단계짜리라 쥔 채로 바로 할 수 있다.
# (B3 은 "서랍을 열고 그 다음 그릇을 넣어라" 라 쥔 채로는 서랍을 못 연다)
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%H:%M:%S) $*"; }

log "[1/3] 핵심 — 쥔 물체가 바로 B 의 목표인 경우 (A2→B9, 현재 0%)"
for g in 50 100 200; do
  $PY switch_experiment.py --task-a 2 --task-b 9 --switch-at grasp:3 \
    --strategy flush --grip-latch "$g" --episodes 10 --out outputs/latch 2>&1 | grep -E "$F"
done

log "[2/3] 해롭지 않은가 — 물체가 필요 없는 B (A8→B0 60%, A8→B7 50%)"
for p in 8:0 8:7; do
  $PY switch_experiment.py --task-a "${p%:*}" --task-b "${p#*:}" --switch-at grasp:3 \
    --strategy flush --grip-latch 100 --episodes 10 --out outputs/latch 2>&1 | grep -E "$F"
done

log "[3/3] A 재개 — 물체를 쥔 채면 돌아갈 수 있나 (지금까지 0/62)"
for p in 8:0 8:7 4:7; do
  $PY switch_experiment.py --task-a "${p%:*}" --task-b "${p#*:}" --switch-at grasp:3 \
    --strategy flush --grip-latch 200 --episodes 10 --out outputs/latch 2>&1 | grep -E "$F"
done

log "끝 — analyze_latch.py 로 분석"
