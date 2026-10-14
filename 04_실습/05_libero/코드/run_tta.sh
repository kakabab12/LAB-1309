#!/usr/bin/env bash
# 10/14 다시 학습하지 않고 바로 시험 (bo1, T0·T8, 장면 1000~1019)
#   g9    : CUDA 그래프 (fastinfer, 출력 같음) → 실제 지연 452ms = 9스텝
#   g7s5  : + 동작 만들기 반복 10→5 → 315ms, 7스텝
#   fn11  : 지연 11, 같은 노이즈를 매 계획마다 (계획이 매번 다른 갈래로 튀는 '맴돌기' 줄이기, 9/18 Panda)
#   g7s5fn: g7s5 + 같은 노이즈
set -u
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 MUJOCO_GL=egl
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
B=${B:-outputs/piper_bo1_model/merged}; N=${N:-bo1}
FN="--switch-noise-seed 0 --noise-seed-from-start --switch-noise-reseed"
ev(){ local o=$1 opt=$2; shift 2; for t in "$@"; do $PY piper_sim/eval_piper.py --policy $B --task-a $t --strategy none --episodes 20 --start-episode 1000 --ttrtc --n-action-steps 1 --cuda-graph $opt --out outputs/piper_${N}${o}_forget 2>&1 | grep -E "$F"; done; }
log "시작 $B"
( ev g9 "--latency-steps 9" 0 8 ) & P1=$!
( ev g7s5 "--latency-steps 7 --num-steps 5" 0 8 ) & P2=$!
( ev fn11 "--latency-steps 11 $FN" 0 8 ) & P3=$!
( ev g7s5fn "--latency-steps 7 --num-steps 5 $FN" 0 8 ) & P4=$!
wait $P1 $P2 $P3 $P4
log "끝"
