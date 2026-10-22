#!/usr/bin/env bash
# 엉뚱한 블록 막기(⑭) 시험 (10/22 16시): 새 기준(grasp:30) 미리 보기와 같은 장면(1000~1004)에서 --wrong-guard 만 더함
cd "$(dirname "$0")"
export PIPER_NOPED=1 PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl
L=outputs/piper/gpu_launch.lock
PD="--approach-assist --descend-assist --unjam-assist --yaw-assist --putdown-assist --fresh-held --putdown-near --wrong-guard"
mkdir -p outputs/piper/claim_gt; rm -rf outputs/piper/claim_gt/*
job(){ local it=$1
  mkdir outputs/piper/claim_gt/$it 2>/dev/null || return 0
  until mkdir $L 2>/dev/null; do sleep 10; done
  ( .venv/bin/python piper_sim/eval_piper.py --policy outputs/piper_np2_model/merged --task-a ${it%:*} --task-b ${it#*:} --switch-at grasp:30 \
      --strategy keep --episodes 5 --start-episode 1000 --latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph $PD \
      --out outputs/piper_guard_test 2>&1 | grep -E "SUMMARY|Traceback|Error:" | grep -v EGL ) & local E=$!
  sleep 60; rmdir $L
  wait $E; }
worker(){ for it in 4:9 8:9 1:7 8:7; do job $it; done; }
( worker ) & W1=$!; sleep 15; ( worker ) & W2=$!
wait $W1 $W2
echo 모두끝
