#!/usr/bin/env bash
# 10/15 정렬 보조(grasp_assist.py) 시험: al1, T0·T8·T5·T6, 지연 9 — 보조 없는 al1 성적(al1h)과 같은 장면 1000~1019
#   al1 28항목이 끝나 교정 시범이 GPU 3개를 쓰기 시작하면, 남은 GPU 메모리가 2.5GB 넘을 때 1개로 (CUDA 그래프 없이 1.75GB)
set -u
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
free_mb(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
until grep -q "^=== .* 끝$" outputs/piper/al_round.log 2>/dev/null; do sleep 120; done
sleep 300
until [ "$(free_mb)" -gt 2500 ]; do sleep 60; done
log "보조 시험 시작 (남은 GPU $(free_mb)MB)"
for t in 0 8 5 6; do
  $PY piper_sim/eval_piper.py --policy outputs/piper_al1_model/merged --task-a $t --strategy none --episodes 20 --start-episode 1000 \
    --latency-steps 9 --ttrtc --n-action-steps 1 --grasp-assist --out outputs/piper_al1assist_forget 2>&1 | grep -E "SUMMARY|Traceback|Error"
done
log "끝"
