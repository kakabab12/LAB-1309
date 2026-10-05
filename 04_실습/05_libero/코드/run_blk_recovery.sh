#!/usr/bin/env bash
# 회복 시범 수집 (10/6 새벽): 빗나가게 쥔·빈손 상태(녹화 안 함)에서 다시 집기·어긋남 보정해 놓기만 녹화
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 MUJOCO_GL=egl
PY=.venv/bin/python; C=piper_sim/collect_recovery.py
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
log "회복 시범 수집 시작"
( $PY $C --tasks 0 1 2 3 --episodes 2600-2679 --seed 71 --out data/piper_blk_recovery > outputs/piper/recovery_1.log 2>&1 ) & P1=$!
( $PY $C --tasks 4 5 6 --episodes 2600-2679 --seed 72 --out data/piper_blk_recovery > outputs/piper/recovery_2.log 2>&1 ) & P2=$!
( $PY $C --tasks 7 8 9 --episodes 2600-2679 --seed 73 --out data/piper_blk_recovery > outputs/piper/recovery_3.log 2>&1 ) & P3=$!
wait $P1 $P2 $P3
$PY piper_sim/trim_stalls.py data/piper_blk_recovery data/piper_blk_recovery_t
log "회복 시범 수집 끝: $(ls data/piper_blk_recovery/episodes | wc -l) — $(grep -h STATS outputs/piper/recovery_*.log | tr '\n' ' ')"
