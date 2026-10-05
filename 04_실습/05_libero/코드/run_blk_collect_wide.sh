#!/usr/bin/env bash
# 넓은 배치 시범 (10/6 아침): 물체를 블록 ±4.5cm·판 ±3.5cm 로 흩어 '평소 자리로 가기' 지름길을 막는다.
#   쥔 손 위치가 실제 블록 위치를 따라가는 기울기: b1c 0.1~0.2, ba1 0.5~0.8 → 물체를 보고 가도록
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 PIPER_BLOCKS_WIDE=1 MUJOCO_GL=egl
PY=.venv/bin/python; C="piper_sim/collect_piper.py"
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
log "넓은 배치 수집 시작"
( $PY $C normal --tasks 0 1 2 3 4 --episodes 2800-2899 --dart 0.1 --seed 81 --out data/piper_blk_wide_normal > outputs/piper/wide_collect_n1.log 2>&1 ) & P1=$!
( $PY $C normal --tasks 5 6 7 8 9 --episodes 2800-2899 --dart 0.1 --seed 82 --out data/piper_blk_wide_normal > outputs/piper/wide_collect_n2.log 2>&1 ) & P2=$!
( $PY $C switch --pairs 8:0,8:3,8:5,8:7,8:9,4:5,4:9,1:7,8:1,8:4,1:8,4:1 --episodes 2800-2849 --dart 0.1 --seed 83 --out data/piper_blk_wide_switch > outputs/piper/wide_collect_s.log 2>&1 ) & P3=$!
wait $P1 $P2 $P3
for d in piper_blk_wide_normal piper_blk_wide_switch; do $PY piper_sim/trim_stalls.py data/$d data/${d}_t; done
log "넓은 배치 수집 끝: 정상 $(ls data/piper_blk_wide_normal/episodes | wc -l), 전환·재개 $(ls data/piper_blk_wide_switch/episodes | wc -l)"
