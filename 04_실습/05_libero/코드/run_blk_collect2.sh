#!/usr/bin/env bash
# 블록 2차 수집 (10/4): 1차 수집이 끝나 학습이 GPU 를 쓰는 동안 CPU 로 시범을 더 모은다 → 2차 학습 데이터.
#   정상 10과제 × 2100~2199, 전환·재개 12쌍 × 2050~2099. 시범 프로그램은 블록 장면에서 거의 100% 라 데이터를 늘리는 값이 싸다.
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
C="piper_sim/collect_piper.py"
until grep -q "블록 수집 끝" outputs/piper/blk_collect.log 2>/dev/null; do sleep 60; done
sleep 600     # 학습이 데이터를 메모리에 올린 뒤에 시작
log "블록 2차 수집 시작"
( $PY $C normal --tasks 0 1 2 3 4 5 6 7 8 9 --episodes 2100-2199 --dart 0.1 --seed 21 --out data/piper_blk_normal2 > outputs/piper/blk2_collect_n.log 2>&1 ) &
( $PY $C switch --pairs 8:0,8:3,8:5,8:7,8:9,4:5 --episodes 2050-2099 --dart 0.1 --seed 23 --out data/piper_blk_switch2 > outputs/piper/blk2_collect_s1.log 2>&1 ) &
( $PY $C switch --pairs 4:9,1:7,8:1,8:4,1:8,4:1 --episodes 2050-2099 --dart 0.1 --seed 24 --out data/piper_blk_switch2 > outputs/piper/blk2_collect_s2.log 2>&1 ) &
wait
for d in piper_blk_normal2 piper_blk_switch2; do $PY piper_sim/trim_stalls.py data/$d data/${d}_t; done
log "블록 2차 수집 끝: 정상 $(ls data/piper_blk_normal2/episodes | wc -l), 전환·재개 $(ls data/piper_blk_switch2/episodes | wc -l)"
