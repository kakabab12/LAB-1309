#!/usr/bin/env bash
# b1c + A2C2 빠른 확인 (10/5 12:50): 빗나감의 주원인이 지연(지연 없으면 0.4cm, 있으면 2~3cm)이라 매 스텝 보정이 맞는 처방인지 b2 전에 확인
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 MUJOCO_GL=egl
PY=.venv/bin/python; F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
POL=outputs/piper_b1c_model/merged
log "A2C2 데이터 (b1c)"
[ -d data/a2c2_piper_b1c ] || $PY a2c2.py gen --policy $POL --ttrtc --data data/piper_blk_normal_t data/piper_blk_switch_t --frac 0.5 --stride 16 --batch 8 --out data/a2c2_piper_b1c 2>&1 | grep -E "끝|Traceback|Error"
log "A2C2 학습"
$PY a2c2.py train --gen data/a2c2_piper_b1c --out outputs/a2c2_piper_b1c --steps 30000 --batch 64 --workers 1 2>&1 | grep -E "끝|val|Traceback|Error" | tail -3
log "b1c + A2C2: T8, T1, A8→B1 (새 장면 1000~1019)"
EV="--episodes 20 --start-episode 1000 --latency-steps 11 --ttrtc --n-action-steps 1 --a2c2 outputs/a2c2_piper_b1c"
( for t in 8 1; do $PY piper_sim/eval_piper.py --policy $POL --task-a $t --strategy none $EV --out outputs/piper_b1cax_forget 2>&1 | grep -E "$F"; done ) & P1=$!
( $PY piper_sim/eval_piper.py --policy $POL --task-a 8 --task-b 1 --switch-at grasp:3 --strategy keep $EV --out outputs/piper_b1cax_switch 2>&1 | grep -E "$F" ) & P2=$!
wait $P1 $P2
log "끝"
