#!/usr/bin/env bash
# CMI 시간 곡선 — 귀는 언제 닫히고 언제 열리는가
#
# 지금은 점 세 개뿐이다: 시작 0.719 / 잡고3 0.284 / 잡고20 0.761
# 사이를 채워야 ① 기제를 확정하고 ② 취약 구간의 길이를 알고
# ③ "열릴 때까지 기다렸다 전환" 이라는 가장 단순한 기준선을 세울 수 있다.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
log(){ echo "=== $(date +%H:%M:%S) $*"; }

if [ $# -ge 1 ]; then
  log "PID $1 대기"
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi

log "CMI 시간 곡선 (잡은 뒤 0·2·4·6·10·15·20·30·40 스텝)"
$PY steerability.py --tasks 8 4 1 2 --episodes 5 --k 8 \
  --probes start,grasp0,grasp2,grasp4,grasp6,grasp10,grasp15,grasp20,grasp30,grasp40 \
  --out outputs/steer_curve 2>&1 | tail -40
$PY cmi_curve.py outputs/steer_curve 2>&1 | tail -25

log "끝"
