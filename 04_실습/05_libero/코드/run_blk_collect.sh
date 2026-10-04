#!/usr/bin/env bash
# PiPER 블록 장면 수집 (10/4): 색 블록·색 판만 있는 장면 (piper_sim/blocks.py).
#   정상 10과제 × 2000~2099 (12:00 에 따로 시작), 전환·재개 12쌍 × 2000~2049. 학습용 장면은 2000번대만.
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
C="piper_sim/collect_piper.py"
log "블록 전환·재개 수집 시작"
( $PY $C switch --pairs 8:0,8:3,8:5,8:7,8:9,4:5 --episodes 2000-2049 --dart 0.1 --seed 13 --out data/piper_blk_switch > outputs/piper/blk_collect_s1.log 2>&1 ) &
( $PY $C switch --pairs 4:9,1:7,8:1,8:4,1:8,4:1 --episodes 2000-2049 --dart 0.1 --seed 14 --out data/piper_blk_switch > outputs/piper/blk_collect_s2.log 2>&1 ) &
wait
while pgrep -f "collect_piper[.]py" >/dev/null; do sleep 30; done      # 정상 수집도 끝날 때까지
for d in piper_blk_normal piper_blk_switch; do $PY piper_sim/trim_stalls.py data/$d data/${d}_t; done
log "블록 수집 끝: 정상 $(ls data/piper_blk_normal/episodes | wc -l), 전환·재개 $(ls data/piper_blk_switch/episodes | wc -l)"
