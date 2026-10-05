#!/usr/bin/env bash
# 블록 4차 수집 (10/5 낮): 집기 전에 일부러 옆으로 최대 2.5cm 빗나갔다가 내려가며 맞추는 시범 (+ 3차의 속도·멈춤 다양화)
#   학생 모델이 지연 때문에 2~3cm 빗나간 채 내려와 손가락이 블록 윗면에 걸렸다 (b1c T8 3/20). 학습 중이라 2프로세스.
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 MUJOCO_GL=egl
PY=.venv/bin/python; C="piper_sim/collect_piper.py"
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
log "블록 4차 수집 시작"
( $PY $C normal --tasks 0 1 2 3 4 5 6 7 8 9 --episodes 2400-2449 --dart 0.1 --seed 41 --out data/piper_blk_normal4 > outputs/piper/blk4_collect_n.log 2>&1 ) & P1=$!
( $PY $C switch --pairs 8:0,8:3,8:5,8:7,8:9,4:5,4:9,1:7,8:1,8:4,1:8,4:1 --episodes 2400-2424 --dart 0.1 --seed 43 --out data/piper_blk_switch4 > outputs/piper/blk4_collect_s.log 2>&1 ) & P2=$!
wait $P1 $P2
for d in piper_blk_normal4 piper_blk_switch4; do $PY piper_sim/trim_stalls.py data/$d data/${d}_t; done
log "블록 4차 수집 끝: 정상 $(ls data/piper_blk_normal4/episodes | wc -l), 전환·재개 $(ls data/piper_blk_switch4/episodes | wc -l)"
