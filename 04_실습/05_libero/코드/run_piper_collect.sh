#!/usr/bin/env bash
# PiPER 시범 수집 (2026-10-03). 학습용 무작위 배치 2000번대만. 점검(audit_piper)이 끝난 뒤 시작.
#   정상 수행: 10개 과제 × 60장면 (2000~2059), 전환·재개: 12쌍 × 30장면 (2000~2029)
#   메모리: 시뮬레이터 프로세스 3개 (각 약 3GB) + 점검 1개 — Panda 학습(13GB)과 같이 돌아도 31GB 안
cd "$(dirname "$0")"
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
log "PiPER 수집 시작"
C="piper_sim/collect_piper.py"
( $PY $C normal --tasks 0 1 2 3 4 5 6 7 8 9 --episodes 2000-2059 --dart 0.1 --out data/piper_normal > outputs/piper/collect_n1.log 2>&1 ) &
( $PY $C switch --pairs 8:0,8:3,8:5,8:7,8:9,4:5 --episodes 2000-2029 --dart 0.1 --out data/piper_switch > outputs/piper/collect_s1.log 2>&1 ) &
( $PY $C switch --pairs 4:9,1:7,8:1,8:4,1:8,4:1 --episodes 2000-2029 --dart 0.1 --out data/piper_switch > outputs/piper/collect_s2.log 2>&1 ) &
wait
log "PiPER 수집 끝: 정상 $(ls data/piper_normal/episodes 2>/dev/null | wc -l), 전환 $(ls data/piper_switch/episodes 2>/dev/null | wc -l)"
