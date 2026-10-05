#!/usr/bin/env bash
# 반복 단계 줄이기 시험 (10/5 14:15): b1c, T8, 새 장면 1000~1019. 지연은 GPU 혼자 잰 생각 시간(50ms=1스텝)으로
#   반복 10 = 522.7ms → 11스텝 (기준: 3/20) / 반복 5 = 354.0ms → 8스텝 / 반복 3 = 288.6ms → 6스텝
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 MUJOCO_GL=egl
PY=.venv/bin/python; F='SUMMARY|Traceback|Error'
POL=outputs/piper_b1c_model/merged
( $PY piper_sim/eval_piper.py --policy $POL --task-a 8 --strategy none --episodes 20 --start-episode 1000 --latency-steps 8 --ttrtc --n-action-steps 1 --num-steps 5 --out outputs/piper_b1c_ns5 2>&1 | grep -E "$F" ) &
( $PY piper_sim/eval_piper.py --policy $POL --task-a 8 --strategy none --episodes 20 --start-episode 1000 --latency-steps 6 --ttrtc --n-action-steps 1 --num-steps 3 --out outputs/piper_b1c_ns3 2>&1 | grep -E "$F" ) &
wait
echo "=== $(date +%m/%d\ %H:%M) 끝"
