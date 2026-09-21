#!/usr/bin/env bash
# 증폭이 정말 "귀를 열어주는가" — CMI 를 증폭 켠 채로 다시 잰다
#
# 왜 필요한가
#   전환 성공률이 안 오르더라도, CMI 가 올랐는지 아닌지로 원인이 갈린다.
#     CMI ↑ 인데 성공 그대로  →  채널은 열렸는데 행동이 안 따라온다 = 채널이 병목이 아니다
#     CMI 그대로              →  우리 증폭이 채널을 실제로 키우지 못한다 = 다른 증폭 방법 필요
#   즉 어느 쪽이든 결론이 난다.
#
# ⚠️ 기준선도 --exclude-own 으로 다시 잰다. 증폭 실험은 A 지시문을 빼고 재므로
#    M 개수(9개)가 같아야 비교가 된다. 9/21 오전의 기준선은 10개라 그대로 못 쓴다.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
log(){ echo "=== $(date +%H:%M:%S) $*"; }

if [ $# -ge 1 ]; then
  log "PID $1 대기"
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi

log "[1/3] 기준선 재측정 (A 지시문 제외, 9개)"
$PY steerability.py --tasks 8 4 1 2 --episodes 5 --k 8 --exclude-own \
  --out outputs/steer_base9 2>&1 | tail -25

log "[2/3] 증폭 w=1.5"
$PY steerability.py --tasks 8 4 1 2 --episodes 5 --k 8 --cfg-w 1.5 \
  --out outputs/steer_cfg15 2>&1 | tail -25

log "[3/3] 증폭 w=3.0 (더 크게 키우면 더 열리는가)"
$PY steerability.py --tasks 8 4 1 2 --episodes 5 --k 8 --cfg-w 3.0 \
  --out outputs/steer_cfg30 2>&1 | tail -25

log "끝 — analyze_steer.py 로 세 폴더를 비교할 것"
