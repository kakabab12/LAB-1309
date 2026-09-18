#!/usr/bin/env bash
# 노이즈 크기(σ)를 바꿔 본다 — "변화가 탈출을 돕는다"의 직접 검증
#
# 근거: σ → 0 (매 추론마다 같은 노이즈) 은 0% 였다. 그러면 σ 를 키우면 더 잘 탈출할까?
#   σ = 0.5  더 결정적 (0 쪽으로)
#   σ = 1.0  원래
#   σ = 1.5, 2.0  더 다양하게
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
W=${1:-0}
log(){ echo "=== $(date +%H:%M:%S) $*"; }
F='SUMMARY|Traceback|Error'
if [ "$W" != "0" ]; then
  log "PID $W 대기"
  while kill -0 "$W" 2>/dev/null; do sleep 120; done
fi
for s in 0.5 1.5 2.0; do
  log "노이즈 크기 σ=$s"
  for p in 8:7 4:7 1:5 2:5; do
    $PY switch_experiment.py --task-a "${p%:*}" --task-b "${p#*:}" --switch-at grasp:3 \
      --strategy flush --noise-scale "$s" --episodes 10 --out outputs/nscale 2>&1 | grep -E "$F"
  done
done
log "끝"
