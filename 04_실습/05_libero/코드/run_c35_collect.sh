#!/usr/bin/env bash
# 3.5cm 블록 시범 수집 (10/7) — run_c35_probe.sh 결과로 작은 블록이 낫다고 나오면 건다.
#   정상 배치 600 + 넓은 배치 600 + 일 바꾸기·돌아오기 12쌍 × 40 + 회복 300 → 끊긴 구간 다듬기 → 위치 검출 미리 계산
set -u
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 PIPER_CUBE=0.035 MUJOCO_GL=egl
PY=.venv/bin/python; C="piper_sim/collect_piper.py"; R=piper_sim/collect_recovery.py
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
PAIRS=8:0,8:3,8:5,8:7,8:9,4:5,4:9,1:7,8:1,8:4,1:8,4:1
log "3.5cm 블록 수집 시작"
( $PY $C normal --tasks 0 1 2 3 4 5 6 7 8 9 --episodes 3100-3159 --dart 0.1 --seed 101 --out data/piper_c35_normal > outputs/piper/c35_collect_n.log 2>&1 ) & P1=$!
( PIPER_BLOCKS_WIDE=1 $PY $C normal --tasks 0 1 2 3 4 5 6 7 8 9 --episodes 3200-3259 --dart 0.1 --seed 102 --out data/piper_c35_wide_normal > outputs/piper/c35_collect_w.log 2>&1 ) & P2=$!
( $PY $C switch --pairs $PAIRS --episodes 3100-3139 --dart 0.1 --seed 103 --out data/piper_c35_switch > outputs/piper/c35_collect_s.log 2>&1
  PIPER_BLOCKS_WIDE=1 $PY $R --tasks 0 1 2 3 4 5 6 7 8 9 --episodes 3300-3329 --seed 104 --out data/piper_c35_recovery > outputs/piper/c35_collect_r.log 2>&1 ) & P3=$!
wait $P1 $P2 $P3
for d in piper_c35_normal piper_c35_wide_normal piper_c35_switch piper_c35_recovery; do $PY piper_sim/trim_stalls.py data/$d data/${d}_t; done
log "3.5cm 수집 끝: 정상 $(ls data/piper_c35_normal/episodes | wc -l), 넓은 $(ls data/piper_c35_wide_normal/episodes | wc -l), 전환·재개 $(ls data/piper_c35_switch/episodes | wc -l), 회복 $(ls data/piper_c35_recovery/episodes | wc -l)"
D="data/piper_c35_normal_t data/piper_c35_wide_normal_t data/piper_c35_switch_t data/piper_c35_recovery_t"
( $PY objfeat_cache.py --data $D --shard 0 --nshard 3 ) & Q1=$!; ( $PY objfeat_cache.py --data $D --shard 1 --nshard 3 ) & Q2=$!; ( $PY objfeat_cache.py --data $D --shard 2 --nshard 3 ) & Q3=$!
wait $Q1 $Q2 $Q3
log "3.5cm 검출 미리 계산 끝"
