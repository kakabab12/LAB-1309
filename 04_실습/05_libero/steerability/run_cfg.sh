#!/usr/bin/env bash
# 지시문 증폭 (instruction CFG) 첫 평가
#
# 근거: CMI 가 물체를 쥔 직후 0.719 → 0.284 (-61%) 로 떨어진다.
#       지시문 채널이 약해지는 것이 문제라면, 그 채널을 키우면 된다.
#       v = v_B + (w-1)(v_B - v_A)   — 이전 지시 A 를 음의 조건으로 쓴다
#
# w=1.0 기준선은 **다시 돌리지 않는다**: w=1 이 원본과 비트 단위로 같음을
#       test_cfg_identity.py 로 확인했으므로 outputs/switch 의 기존 결과가 곧 기준선이다.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%H:%M:%S) $*"; }

if [ $# -ge 1 ]; then
  log "PID $1 대기"
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi

# 잡은 직후(grasp:3) — CMI 가 가장 낮은 지점이라 효과가 있다면 여기서 나와야 한다
for w in 1.5 2.0 3.0; do
  log "[증폭 w=$w] 잡은 직후"
  for p in 8:7 4:7 1:5 2:5; do
    $PY switch_experiment.py --task-a "${p%:*}" --task-b "${p#*:}" --switch-at grasp:3 \
      --strategy flush --cfg-w "$w" --episodes 10 --out outputs/cfg 2>&1 | grep -E "$F"
  done
done

# 가장 좋았던 w 는 월요일에 사람이 보고 정한다. 들고 이동 중은 그 다음.
log "끝 — analyze_cfg.py 로 분석"
