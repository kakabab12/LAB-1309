#!/usr/bin/env bash
# 정렬 보조 시험 (al1, 지연 9, CUDA 그래프) — 보조 없는 al1h 와 같은 장면 1000~1019
set -u
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
ev(){ for t in "$@"; do $PY piper_sim/eval_piper.py --policy outputs/piper_al1_model/merged --task-a $t --strategy none --episodes 20 --start-episode 1000 \
    --latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph --grasp-assist --out outputs/piper_al1assist_forget 2>&1 | grep -E "SUMMARY|Traceback|Error"; done; }
evp(){ for p in "$@"; do $PY piper_sim/eval_piper.py --policy outputs/piper_al1_model/merged --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy keep --episodes 20 --start-episode 1000 \
    --latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph --grasp-assist --out outputs/piper_al1assist_switch 2>&1 | grep -E "SUMMARY|Traceback|Error"; done; }
log "시작"
( ev 0 5 ) & P1=$!; ( ev 8 6 ) & P2=$!; ( evp 8:5 ) & P3=$!; ( evp 4:9 ) & P4=$!
wait $P1 $P2 $P3 $P4
log "끝"
