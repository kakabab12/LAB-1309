#!/usr/bin/env bash
# alr1 + 정렬 보조 28항목 (10/15): alr1 보조 없는 28항목이 끝나면, GPU 메모리가 5GB 넘게 남을 때 2개로
set -u
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 MUJOCO_GL=egl
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
free_mb(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph --grasp-assist"
B=outputs/piper_alr1_model/merged; o=piper_alr1assist
ev_t(){ for t in "$@"; do $PY piper_sim/eval_piper.py --policy $B --task-a $t --strategy none --episodes 20 --start-episode 1000 $EVQ --out outputs/${o}_forget 2>&1 | grep -E "$F"; done; }
ev_p(){ for p in "$@"; do $PY piper_sim/eval_piper.py --policy $B --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy keep --episodes 20 --start-episode 1000 $EVQ --out outputs/${o}_switch 2>&1 | grep -E "$F"; done; }
until grep -q "score_piper_alr1h_loose" outputs/piper/alr_round.log 2>/dev/null; do sleep 120; done
until [ "$(free_mb)" -gt 5000 ]; do sleep 60; done
log "alr1 + 정렬 보조 28항목 시작"
( ev_t 0 1 2 3 4 5 6 7 8 9 ) & P1=$!; ( ev_p 8:0 8:3 8:5 8:7 8:9 4:5 4:9 1:7 8:1 8:4 1:8 4:1 ) & P2=$!
wait $P1 $P2
$PY scoreboard.py --blocks --forget outputs/${o}_forget --switch outputs/${o}_switch --name "alr1 + 정렬 보조 (지연 9)" --png outputs/media/score_${o}.png --md outputs/piper/score_${o}.md | tail -1
$PY scoreboard.py --blocks --loose --forget outputs/${o}_forget --switch outputs/${o}_switch --name "alr1 + 정렬 보조 판 안 기준" --png outputs/media/score_${o}_loose.png --md outputs/piper/score_${o}_loose.md | tail -1
log "끝"
