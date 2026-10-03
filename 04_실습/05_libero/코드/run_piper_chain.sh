#!/usr/bin/env bash
# PiPER 자동 라운드 (10/3): 1차 학습·평가가 끝나면 s1→s2, s2→s3 (교정 시범 장면은 라운드마다 새 배치)
cd "$(dirname "$0")"
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
until grep -qE "^=== .* 끝$|멈춤" outputs/piper/run_train.log 2>/dev/null; do sleep 120; done
grep -q "멈춤" outputs/piper/run_train.log && { log "1차가 멈춤 — 체인 멈춤"; exit 1; }
while pgrep -f "collect_piper[.]py" >/dev/null; do sleep 60; done          # 2차 수집이 끝나야 학습 데이터에 들어간다
log "라운드 s1 → s2"
./run_piper_round.sh piper_s1 piper_s2 2100-2129 > outputs/piper/round_s2.log 2>&1
grep -q "라운드 끝" outputs/piper/round_s2.log || { log "s2 라운드 멈춤"; exit 1; }
log "라운드 s2 → s3"
./run_piper_round.sh piper_s2 piper_s3 2130-2159 > outputs/piper/round_s3.log 2>&1
log "끝"
