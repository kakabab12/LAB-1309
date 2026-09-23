#!/usr/bin/env bash
# ⚠️ **진단 실험** — 정렬 붕괴가 집기 실패의 '원인'인가, '증상'인가
#
# 측정 (2026-09-23)
#   그리퍼가 닫히는 순간의 손목 이탈:
#       교란 없이 성공한 집기   6.6도
#       교란 27cm              22~28도  (3~4배)
#   정렬이 무너지는 만큼 성공률이 떨어진다 (100 → 0%).
#   하지만 이건 **상관**이다. 같이 나빠지는 증상일 수도 있다.
#
# 이 실험
#   손목만 초기 방향 쪽으로 조금씩 당겨 본다 (--wrist-bias).
#   위치는 건드리지 않고, 스크립트 복귀도 아니고, 매 스텝 조금씩만 당긴다.
#
#     집기가 살아나면  → 정렬이 **원인**이다. 학습이 겨냥할 곳이 맞다
#     안 살아나면     → 정렬은 **증상**이다. 다른 것을 찾아야 한다
#
# ⚠️ 회전을 건드리므로 제약(초기 자세 복귀 금지)과 **회색지대**다.
#    `ret_rot`(회전 복귀)은 이미 제약 위반으로 분류했다.
#    이건 진단용이며, 방법으로 삼으려면 자연스러움 지표를 따로 봐야 한다.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%H:%M:%S) $*"; }

if [ $# -ge 1 ]; then
  log "PID $1 대기"
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi

# B 가 물체인 쌍(현재 0%)에서 본다. 여기서 안 오르면 정렬은 원인이 아니다.
for k in 0.05 0.15; do
  log "손목 보정 k=$k"
  for p in 8:3 4:9 8:5; do
    $PY switch_experiment.py --task-a "${p%:*}" --task-b "${p#*:}" --switch-at grasp:3 \
      --strategy flush --wrist-bias "$k" --episodes 10 --out outputs/wrist 2>&1 | grep -E "$F"
  done
done

log "대조 — 가구·기구 B 에는 해롭지 않은가 (현재 60/50%)"
for p in 8:0 8:7; do
  $PY switch_experiment.py --task-a "${p%:*}" --task-b "${p#*:}" --switch-at grasp:3 \
    --strategy flush --wrist-bias 0.15 --episodes 10 --out outputs/wrist 2>&1 | grep -E "$F"
done

log "끝 — 자연스러움도 함께 볼 것 (jerk, 초기자세까지 거리)"
