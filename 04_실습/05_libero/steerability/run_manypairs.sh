#!/usr/bin/env bash
# 쌍을 4개 → 20개로 늘린다
#
# 왜 필요한가 (두 문제를 한 번에 푼다)
#   ① 표본: 쌍당 10회 × 4쌍 = 36회로는 **20%p 미만을 못 본다**.
#      20쌍이면 군당 180~200회가 되어 **10%p 대까지** 볼 수 있다
#   ② 쌍 난이도: 지금 8→7 50% / 1→5 0% 처럼 갈리는데 이유를 모른다.
#      쌍별 조종 가능성(분리도)이 이걸 예측하는지 보려면 쌍이 많아야 한다
#
# A 는 물체를 쥐는 태스크만 쓴다 (grasp:3 에서 전환해야 하므로): 8·4·1·2
# B 는 다양하게: 0(서랍) 3(서랍+그릇) 5(접시 밀기) 7(스토브) 9(와인→선반)
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%H:%M:%S) $*"; }

if [ $# -ge 1 ]; then
  log "PID $1 대기"
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi

log "기준선(flush)을 20쌍으로 확대 — 이미 있는 4쌍은 건너뛴다"
for a in 8 4 1 2; do
  for b in 0 3 5 7 9; do
    [ "$a" = "$b" ] && continue
    out="outputs/switch/A${a}_B${b}_grasp3_flush.json"
    if [ -f "$out" ]; then
      log "  A$a→B$b 는 이미 있음, 건너뜀"
      continue
    fi
    $PY switch_experiment.py --task-a "$a" --task-b "$b" --switch-at grasp:3 \
      --strategy flush --episodes 10 --out outputs/switch 2>&1 | grep -E "$F"
  done
done

log "쌍별 조종 가능성이 성공률을 예측하는지 검정"
$PY pair_steer.py outputs/steer_curve 2>&1 | tail -30

log "끝"
