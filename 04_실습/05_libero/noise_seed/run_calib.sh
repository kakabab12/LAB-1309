#!/usr/bin/env bash
# "장면마다 노이즈를 보정하면 되는가" 검증
#
# 배경: 좋은 시드가 처음 보는 쌍(6→0)에서는 80%→20% 로 해로웠다. 하지만 **고를 때 쓴 쌍**에서는 +30%p 였다.
#       그러면 "보편적으로 좋은 노이즈"는 없지만 **장면마다 보정하면 된다**일 수 있다.
#
# 설계: 6→0 쌍에서
#   1) 에피소드 0~9 로 오라클(시드 8개) → 그 쌍에 좋은 시드를 고른다
#   2) 에피소드 10~19 (미사용)에서 기본 vs 그 시드 비교
#   같은 절차를 9→7 에도 (기본 0% 라 바닥 효과가 있을 수 있음)
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
W=${1:-0}
log(){ echo "=== $(date +%H:%M:%S) $*"; }
F='SUMMARY|Traceback|Error'
if [ "$W" != "0" ]; then
  log "PID $W 대기"
  while kill -0 "$W" 2>/dev/null; do sleep 60; done
fi

log "[1/2] 6→0 쌍 전용 오라클 (에피소드 0~9, 시드 8개)"
$PY oracle_bon.py --pairs 6:0 --episodes 10 --n 8 --out outputs/oracle_60 2>&1 | tail -12
$PY analyze_oracle.py outputs/oracle_60 2>&1 | tail -16

BEST=$($PY - <<'PY'
import json
try:
    ps = json.load(open("outputs/report/oracle_60.json")).get("per_seed") or []
    print(10000 + 97 * max(range(len(ps)), key=lambda i: ps[i]) if ps else -1)
except Exception:
    print(-1)
PY
)
log "6→0 에 좋은 시드: $BEST"
if [ "$BEST" != "-1" ]; then
  log "[2/2] 미사용 에피소드(10~19)에서 검증"
  $PY switch_experiment.py --task-a 6 --task-b 0 --switch-at grasp:3 --strategy flush \
    --episodes 10 --start-episode 10 --out outputs/calib 2>&1 | grep -E "$F"
  $PY switch_experiment.py --task-a 6 --task-b 0 --switch-at grasp:3 --strategy flush \
    --switch-noise-seed "$BEST" --episodes 10 --start-episode 10 --out outputs/calib 2>&1 | grep -E "$F"
  $PY analyze_seedtest.py outputs/calib 2>&1 | tail -14
fi
log "끝"
