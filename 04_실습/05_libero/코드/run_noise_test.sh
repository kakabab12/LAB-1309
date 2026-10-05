#!/usr/bin/env bash
# 잡음 0 시험 (10/6 아침): 절대 목표 모델 ba1 이 덜덜 떨림(흔들림 25~30, 변화량 모델 12) — 계획마다 출발 잡음이 달라 목표가 흔들린다는 가설
#   같은 T2 장면 1000~1019 에서 ba1 (잡음 1: 3cm 2/20, 판 안 8/20) 과 비교
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 MUJOCO_GL=egl
.venv/bin/python piper_sim/eval_piper.py --policy outputs/piper_ba1_model/merged --task-a 2 --strategy none --episodes 20 --start-episode 1000 \
  --latency-steps 11 --ttrtc --n-action-steps 1 --noise-scale 0 --noise-seed-from-start --out outputs/piper_ba1_n0 2>&1 | grep -E "SUMMARY|Traceback|Error"
echo "=== $(date +%m/%d\ %H:%M) 끝"
