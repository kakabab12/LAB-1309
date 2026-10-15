#!/usr/bin/env bash
# 놓기 보조 시험 (10/16): al1 + 정렬 + 놓기 보조, 정렬 보조 시험(piper_al1assist)과 같은 8항목·장면 1000~1019, 지연 9
#   als1 28항목이 끝나 교정 시범(GPU 3개)이 시작되면 남은 GPU 메모리 2.5GB 넘을 때 1개로 (CUDA 그래프 없이)
set -u
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
free_mb(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
Q="--latency-steps 9 --ttrtc --n-action-steps 1 --place-assist"
until grep -q "score_piper_als1h_loose" outputs/piper/als_round.log 2>/dev/null; do sleep 120; done
sleep 600
until [ "$(free_mb)" -gt 2500 ]; do sleep 60; done
log "놓기 보조 시험 시작 (남은 GPU $(free_mb)MB)"
for t in 0 8 5 6; do
  $PY piper_sim/eval_piper.py --policy outputs/piper_al1_model/merged --task-a $t --strategy none --episodes 20 --start-episode 1000 $Q --out outputs/piper_al1place_forget 2>&1 | grep -E "SUMMARY|Traceback|Error"
done
for p in 8:5 4:9; do
  $PY piper_sim/eval_piper.py --policy outputs/piper_al1_model/merged --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy keep --episodes 20 --start-episode 1000 $Q --out outputs/piper_al1place_switch 2>&1 | grep -E "SUMMARY|Traceback|Error"
done
log "끝"
