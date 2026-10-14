#!/usr/bin/env bash
# 10/14: 지연이 얼마나 깎나 — bo1 을 지연 0·3·6 스텝으로 (11 스텝 결과는 outputs/piper_bo1h_forget 에 있음)
#   + 작은 블록 3.5cm 시험 (지연 11, 다시 학습 안 함). 같은 장면 1000~1019, T0·T8
set -u
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 MUJOCO_GL=egl
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
B=outputs/piper_bo1_model/merged
ev(){ local o=$1 lat=$2; shift 2; for t in "$@"; do $PY piper_sim/eval_piper.py --policy $B --task-a $t --strategy none --episodes 20 --start-episode 1000 --latency-steps $lat --ttrtc --n-action-steps 1 --out outputs/${o}_forget 2>&1 | grep -E "$F"; done; }
log "시작"
( ev piper_bo1lat0 0 0 8 ) & P1=$!
( ev piper_bo1lat3 3 0 8 ) & P2=$!
( ev piper_bo1lat6 6 0 8 ) & P3=$!
( PIPER_CUBE=0.035 ev piper_bo1c35 11 0 8 ) & P4=$!
wait $P1 $P2 $P3 $P4
log "끝"
