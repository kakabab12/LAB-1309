#!/usr/bin/env bash
# 새 쥐기 방식(align) 시범 수집 (10/14): 블록 위에서 먼저 맞추고 똑바로, 마지막 3cm 천천히, 3~6스텝 멈췄다 쥠.
#   TAG·CUBE 로 블록 크기를 고른다:  TAG=al CUBE=0.045 ./run_align_collect.sh   /  TAG=al35 CUBE=0.035 ...
#   정상 배치 600 + 넓은 배치 600 + 일 바꾸기·돌아오기 12쌍 x 40 + 회복 300 → 다듬기 → 위치 검출 미리 계산
set -u
cd "$(dirname "$0")"
TAG=${TAG:-al}; CUBE=${CUBE:-0.045}
export PIPER_BLOCKS=1 PIPER_CUBE=$CUBE PIPER_GRASP_STYLE=align MUJOCO_GL=egl
PY=.venv/bin/python; C="piper_sim/collect_piper.py"; R=piper_sim/collect_recovery.py
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
PAIRS=8:0,8:3,8:5,8:7,8:9,4:5,4:9,1:7,8:1,8:4,1:8,4:1
log "$TAG 수집 시작 (블록 ${CUBE}m, 쥐기 align)"
( $PY $C normal --tasks 0 1 2 3 4 5 6 7 8 9 --episodes 3100-3159 --dart 0.1 --seed 111 --out data/piper_${TAG}_normal > outputs/piper/${TAG}_collect_n.log 2>&1 ) & P1=$!
( PIPER_BLOCKS_WIDE=1 $PY $C normal --tasks 0 1 2 3 4 5 6 7 8 9 --episodes 3200-3259 --dart 0.1 --seed 112 --out data/piper_${TAG}_wide_normal > outputs/piper/${TAG}_collect_w.log 2>&1 ) & P2=$!
( $PY $C switch --pairs $PAIRS --episodes 3100-3139 --dart 0.1 --seed 113 --out data/piper_${TAG}_switch > outputs/piper/${TAG}_collect_s.log 2>&1
  PIPER_BLOCKS_WIDE=1 $PY $R --tasks 0 1 2 3 4 5 6 7 8 9 --episodes 3300-3329 --seed 114 --out data/piper_${TAG}_recovery > outputs/piper/${TAG}_collect_r.log 2>&1 ) & P3=$!
wait $P1 $P2 $P3
for d in normal wide_normal switch recovery; do $PY piper_sim/trim_stalls.py data/piper_${TAG}_$d data/piper_${TAG}_${d}_t; done
log "$TAG 수집 끝: 정상 $(ls data/piper_${TAG}_normal/episodes | wc -l), 넓은 $(ls data/piper_${TAG}_wide_normal/episodes | wc -l), 전환·재개 $(ls data/piper_${TAG}_switch/episodes | wc -l), 회복 $(ls data/piper_${TAG}_recovery/episodes | wc -l)"
D="data/piper_${TAG}_normal_t data/piper_${TAG}_wide_normal_t data/piper_${TAG}_switch_t data/piper_${TAG}_recovery_t"
( $PY objfeat_cache.py --data $D --shard 0 --nshard 3 ) & Q1=$!; ( $PY objfeat_cache.py --data $D --shard 1 --nshard 3 ) & Q2=$!; ( $PY objfeat_cache.py --data $D --shard 2 --nshard 3 ) & Q3=$!
wait $Q1 $Q2 $Q3
log "$TAG 검출 미리 계산 끝"
