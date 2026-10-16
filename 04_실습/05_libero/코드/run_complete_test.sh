#!/usr/bin/env bash
# 쥐기 마무리 시험 (10/16): als1 + 정렬·놓기 보조 vs + 쥐기 마무리, T0·T8·A8→B5·A4→B9, 장면 1000~1019, 지연 9
set -u
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
B=outputs/piper_als1_model/merged
Q="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
run(){ local o=$1 opt=$2
  for t in 0 8; do $PY piper_sim/eval_piper.py --policy $B --task-a $t --strategy none --episodes 20 --start-episode 1000 $Q $opt --out outputs/${o}_forget 2>&1 | grep -E "SUMMARY|Traceback|Error"; done
  for p in 8:5 4:9; do $PY piper_sim/eval_piper.py --policy $B --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy keep --episodes 20 --start-episode 1000 $Q $opt --out outputs/${o}_switch 2>&1 | grep -E "SUMMARY|Traceback|Error"; done; }
log "시작"
( run piper_als1place "--place-assist" ) & P1=$!
( run piper_als1comp "--grasp-complete" ) & P2=$!
wait $P1 $P2
log "끝"
