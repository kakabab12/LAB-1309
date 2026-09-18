#!/usr/bin/env bash
# 노이즈 평균 이동을 CEM 으로 최적화하고, **미사용 에피소드**에서 검증한다.
#
# 왜 이 형태인가 (2026-09-18 측정 근거)
#   · 노이즈만 바꿔도 성공률이 양방향 60~70%p 움직인다  → 조종 손잡이로 쓸 수 있다
#   · 고정된 좋은 시드는 장면마다 달라 일반화 안 된다      → 장면마다 찾아야 한다
#   · 매 추론마다 같은 노이즈를 쓰면 0% 다                 → 고정하지 말고 **분포 중심만** 옮긴다
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

for p in 8:7 6:0; do
  A=${p%:*}; B=${p#*:}
  log "CEM 최적화 A$A→B$B (에피소드 0~5 로 찾기)"
  $PY cem_noise.py --task-a "$A" --task-b "$B" --iters 8 --candidates 6 --elite 2 \
    --episodes 6 --start-episode 0 --out "outputs/cem_${A}_${B}" 2>&1 | tail -30

  MU="outputs/cem_${A}_${B}/mu_best.npy"
  if [ -f "$MU" ]; then
    log "검증 A$A→B$B (미사용 에피소드 10~19)"
    $PY switch_experiment.py --task-a "$A" --task-b "$B" --switch-at grasp:3 --strategy flush \
      --episodes 10 --start-episode 10 --out outputs/cemtest 2>&1 | grep -E "$F"
    $PY switch_experiment.py --task-a "$A" --task-b "$B" --switch-at grasp:3 --strategy flush \
      --noise-shift "$MU" --episodes 10 --start-episode 10 --out outputs/cemtest 2>&1 | grep -E "$F"
  fi
done
log "끝"
