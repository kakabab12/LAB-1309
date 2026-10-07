#!/usr/bin/env bash
# 넓은 배치 회복 시범 (10/6 낮, 다음 라운드용): 물체를 넓게 흩은 장면에서 빗나간·어긋나게 쥔 상태부터 다시 집기·보정해 놓기
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 PIPER_BLOCKS_WIDE=1 MUJOCO_GL=egl
PY=.venv/bin/python; C=piper_sim/collect_recovery.py
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
log "넓은 배치 회복 시범 시작"
( $PY $C --tasks 0 1 2 3 --episodes 2900-2949 --seed 91 --out data/piper_blk_wide_recovery > outputs/piper/wrec_1.log 2>&1 ) & P1=$!
( $PY $C --tasks 4 5 6 --episodes 2900-2949 --seed 92 --out data/piper_blk_wide_recovery > outputs/piper/wrec_2.log 2>&1 ) & P2=$!
( $PY $C --tasks 7 8 9 --episodes 2900-2949 --seed 93 --out data/piper_blk_wide_recovery > outputs/piper/wrec_3.log 2>&1 ) & P3=$!
wait $P1 $P2 $P3
$PY piper_sim/trim_stalls.py data/piper_blk_wide_recovery data/piper_blk_wide_recovery_t
log "넓은 배치 회복 시범 끝: $(ls data/piper_blk_wide_recovery/episodes | wc -l)"
