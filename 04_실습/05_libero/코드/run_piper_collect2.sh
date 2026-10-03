#!/usr/bin/env bash
# PiPER 시범 2차 수집 (10/3 저녁): 고친 시범 프로그램(멈춤 끊기, 놓을 자리까지 보는 쥐기, 팔 충돌 검사, 선반 뒤 준비 자세)
#   정상 10과제 × 2060~2099, 전환·재개 12쌍 × 2030~2059. 프로세스 2개 (PiPER 학습·Panda 기록 평가와 같이 돌아서)
cd "$(dirname "$0")"
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
while pgrep -f "audit_piper[.]py" >/dev/null; do sleep 60; done
log "PiPER 2차 수집 시작"
C="piper_sim/collect_piper.py"
( $PY $C normal --tasks 0 1 2 3 4 5 6 7 8 9 --episodes 2060-2099 --dart 0.1 --seed 1 --out data/piper_normal2 > outputs/piper/collect2_n.log 2>&1 ) &
( $PY $C switch --pairs 8:0,8:3,8:5,8:7,8:9,4:5,4:9,1:7,8:1,8:4,1:8,4:1 --episodes 2030-2059 --dart 0.1 --seed 1 --out data/piper_switch2 > outputs/piper/collect2_s.log 2>&1 ) &
wait
for d in piper_normal2 piper_switch2; do $PY piper_sim/trim_stalls.py data/$d data/${d}_t; done
log "PiPER 2차 수집 끝: 정상 $(ls data/piper_normal2/episodes | wc -l), 전환 $(ls data/piper_switch2/episodes | wc -l)"
