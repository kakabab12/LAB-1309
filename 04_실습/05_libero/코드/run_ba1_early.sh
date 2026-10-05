#!/usr/bin/env bash
# ba1 첫 저장본(4000스텝)으로 T8 미리 보기 (10/5 23시): 절대 목표 방식이 맞는 방향인지 학습 끝나기 전에 확인
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 MUJOCO_GL=egl
until [ -f outputs/piper_ba1_model/step_4000/abs_pos ] && [ -f outputs/piper_ba1_model/step_4000/model.safetensors ]; do sleep 60; done
sleep 60
.venv/bin/python piper_sim/eval_piper.py --policy outputs/piper_ba1_model/step_4000 --task-a 8 --strategy none --episodes 10 --start-episode 1000 \
  --latency-steps 11 --ttrtc --n-action-steps 1 --out outputs/piper_ba1_s4k 2>&1 | grep -E "SUMMARY|Traceback|Error"
echo "=== $(date +%m/%d\ %H:%M) 끝"
