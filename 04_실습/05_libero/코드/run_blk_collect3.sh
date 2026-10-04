#!/usr/bin/env bash
# 블록 3차 수집 (10/4 밤, CPU 가 차서 절반으로 줄임: 정상 50장면, 쌍마다 25장면): 내려가는 속도·바닥에서 멈춤을 시범마다 다르게 한 시범 프로그램으로 (시간으로 오므리는 버릇 막기)
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 MUJOCO_GL=egl
PY=.venv/bin/python; C="piper_sim/collect_piper.py"
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
log "블록 3차 수집 시작"
( $PY $C normal --tasks 0 1 2 3 4 5 6 7 8 9 --episodes 2300-2349 --dart 0.1 --seed 31 --out data/piper_blk_normal3 > outputs/piper/blk3_collect_n.log 2>&1 ) &
( $PY $C switch --pairs 8:0,8:3,8:5,8:7,8:9,4:5 --episodes 2300-2324 --dart 0.1 --seed 33 --out data/piper_blk_switch3 > outputs/piper/blk3_collect_s1.log 2>&1 ) &
( $PY $C switch --pairs 4:9,1:7,8:1,8:4,1:8,4:1 --episodes 2300-2324 --dart 0.1 --seed 34 --out data/piper_blk_switch3 > outputs/piper/blk3_collect_s2.log 2>&1 ) &
wait
for d in piper_blk_normal3 piper_blk_switch3; do $PY piper_sim/trim_stalls.py data/$d data/${d}_t; done
log "블록 3차 수집 끝: 정상 $(ls data/piper_blk_normal3/episodes | wc -l), 전환·재개 $(ls data/piper_blk_switch3/episodes | wc -l)"
