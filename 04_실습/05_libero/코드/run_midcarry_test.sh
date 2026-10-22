#!/usr/bin/env bash
# 옮기는 도중 새 지시 시험 (10/22 15시): 사용자 "지금 바로 쥐었다가 바로 놓잖아 — 어느 정도 수행했다가 다른 곳에 놨다가 다른 태스크 하고
#   다시 하는 걸로 못 하나". 새 지시를 쥔 뒤 3스텝(0.15초) 대신 30스텝(1.5초, 들고 옮기는 중)에 준다. 내려놓기는 손에서 가장 가까운 빈 자리
#   (--putdown-near), 쥔 블록 알아보기 (--fresh-held). 돌아오기 6쌍 × 장면 1000~1004. 학습 옆이라 일꾼 2개
cd "$(dirname "$0")"
export PIPER_NOPED=1 PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl
L=outputs/piper/gpu_launch.lock
while kill -0 3056100 2>/dev/null; do sleep 20; done          # 쥔 블록 알아보기 시험 끝 기다림
PD="--approach-assist --descend-assist --unjam-assist --yaw-assist --putdown-assist --fresh-held --putdown-near"
mkdir -p outputs/piper/claim_mc; rm -rf outputs/piper/claim_mc/*
job(){ local it=$1
  mkdir outputs/piper/claim_mc/$it 2>/dev/null || return 0
  until mkdir $L 2>/dev/null; do sleep 10; done
  ( .venv/bin/python piper_sim/eval_piper.py --policy outputs/piper_np2_model/merged --task-a ${it%:*} --task-b ${it#*:} --switch-at grasp:30 \
      --strategy keep --episodes 5 --start-episode 1000 --latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph $PD \
      --out outputs/piper_midcarry_test 2>&1 | grep -E "SUMMARY|Traceback|Error:" | grep -v EGL ) & local E=$!
  sleep 90; rmdir $L
  wait $E; }
worker(){ for it in 8:5 8:7 8:9 4:5 4:9 1:7; do job $it; done; }
( worker ) & W1=$!; sleep 20; ( worker ) & W2=$!
wait $W1 $W2
echo 모두끝
