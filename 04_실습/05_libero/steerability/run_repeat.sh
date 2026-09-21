#!/usr/bin/env bash
# 대안 ① 지시문 토큰 반복 — 가장 단순한 증폭
#
# 왜: CMI 가 쥔 직후 0.284 로 떨어지는 것이 "언어가 밀려나서" 라면,
#     같은 말을 여러 번 넣어 언어 토큰의 비중을 키우는 것만으로 달라질 수 있다.
#     속도장 증폭과 달리 **동작 크기가 안 커져** 자연스러움에 안전하고,
#     추론도 한 번뿐이라 지연 문제도 없다.
#
# 먼저 CMI 로 "귀가 열리는지" 보고, 열리면 전환 성공률을 잰다.
# (성공률은 표본이 비싸니, 기제가 확인된 뒤에 돌린다)
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%H:%M:%S) $*"; }

if [ $# -ge 1 ]; then
  log "PID $1 대기"
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi

log "[1/2] 반복이 CMI 를 올리는가 (k=3)"
$PY steerability.py --tasks 8 4 1 2 --episodes 5 --k 8 \
  --probes start,grasp3,grasp20 --instr-repeat 3 \
  --out outputs/steer_rep3 2>&1 | tail -25
$PY compare_steer.py outputs/steerability outputs/steer_rep3 2>&1 | tail -20

log "[2/2] 전환 성공률 (k=3)"
for p in 8:7 4:7 1:5 2:5; do
  $PY switch_experiment.py --task-a "${p%:*}" --task-b "${p#*:}" --switch-at grasp:3 \
    --strategy flush --instr-repeat 3 --episodes 10 --out outputs/cfg 2>&1 | grep -E "$F"
done

log "끝"
