#!/usr/bin/env bash
# 쥔 블록 알아보기(⑬) 시험 (10/22 14시): np2 + 보조 5가지 (93.8%) 의 돌아오기 실패 장면 (원래 블록을 판 밖에) 을 --fresh-held 로 그대로 다시
#   + 원래 성공한 장면 2개 (망가지지 않나). 학습 옆이라 하나씩, 자물쇠 잡고
cd "$(dirname "$0")"
export PIPER_NOPED=1 PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl
L=outputs/piper/gpu_launch.lock
PD="--approach-assist --descend-assist --unjam-assist --yaw-assist --putdown-assist --fresh-held"
for it in 1:7:1001 1:7:1004 8:7:1008 8:7:1014 8:9:1003 8:9:1017 4:9:1016 1:7:1006 1:7:1017 8:5:1000 4:5:1001; do
  IFS=: read a b ep <<< "$it"
  until mkdir $L 2>/dev/null; do sleep 5; done
  .venv/bin/python piper_sim/eval_piper.py --policy outputs/piper_np2_model/merged --task-a $a --task-b $b --switch-at grasp:3 --strategy keep \
    --episodes 1 --start-episode $ep --latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph $PD --out outputs/piper_fh_test/A${a}B${b}_$ep 2>&1 | grep -E "Traceback|Error:" | grep -v EGL | head -2
  rmdir $L
  echo "$(date +%H:%M) 끝 A${a}B${b} $ep"
done
echo 모두끝
