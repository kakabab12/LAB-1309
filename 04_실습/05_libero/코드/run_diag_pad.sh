#!/usr/bin/env bash
# 판 밀림 진단 (10/22 밤): 새 기준 돌아오기 실패 장면을 같은 장면 번호로 다시 돌려 판이 언제·무엇에 밀리나 본다.
#   평가 일꾼 하나(3295053)를 멈춰 비운 GPU 자리 하나만 쓴다 — 그 일꾼의 마지막 평가(3295063)가 끝나면 시작
cd "$(dirname "$0")"
export PIPER_NOPED=1 PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl
L=outputs/piper/gpu_launch.lock
BEST="--approach-assist --descend-assist --unjam-assist --yaw-assist --putdown-assist --fresh-held --putdown-near --wrong-guard --release-wait 40"
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph --switch-at grasp:30 --strategy keep --policy outputs/piper_np2_model/merged"
while kill -0 3295063 2>/dev/null; do sleep 20; done
for it in 8:7:1005 4:9:1000 4:9:1004 8:7:1000 8:7:1009; do
  IFS=: read a b e <<< "$it"
  until mkdir $L 2>/dev/null; do sleep 10; done
  ( sleep 60; rmdir $L ) &
  echo "=== A$a B$b 장면 $e"
  .venv/bin/python diag_pad.py -- --task-a $a --task-b $b --episodes 1 --start-episode $e $EVQ $BEST --out outputs/piper_diag_pad 2>&1 \
    | grep -E "^\[|^   \(|SUMMARY|Traceback|Error:" | grep -v EGL
  wait
done
echo 끝
