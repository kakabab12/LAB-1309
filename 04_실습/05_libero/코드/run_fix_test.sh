#!/usr/bin/env bash
# 고친 보조 시험 (10/22 밤): 새 기준 돌아오기·B 실패 장면을 같은 장면 번호로 다시 (같은 장면 = 같은 결과라 바뀐 것만 효과).
#   고친 것 = 넓은 변환식 + 쥐기 확정(⑩) + 판 지키기(⑮) + 판 따라가기(⑯) + 기억 고치기(⑰). 평가 일꾼 하나를 멈춰 비운 GPU 자리 하나만 쓴다
cd "$(dirname "$0")"
export PIPER_NOPED=1 PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl PIPER_FRONT_CALIB=piper_sim/front_calib_wide.json
L=outputs/piper/gpu_launch.lock
BEST="--approach-assist --descend-assist --unjam-assist --yaw-assist --putdown-assist --fresh-held --putdown-near --wrong-guard --release-wait 40"
FIX="--close-assist --pad-safe --pad-track --mem-refresh"
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph --switch-at grasp:30 --strategy keep --policy outputs/piper_np2_model/merged"
ITEMS="4:9:1000 4:9:1004 8:7:1000 8:7:1005 8:7:1006 8:7:1009 8:7:1010 8:7:1014 8:7:1017 8:7:1019 8:5:1006 8:5:1008 8:5:1013 8:5:1016 8:5:1017 8:0:1005 8:0:1006 8:0:1009 8:0:1010 8:0:1019"
for it in $ITEMS; do
  IFS=: read a b e <<< "$it"
  until mkdir $L 2>/dev/null; do sleep 10; done
  ( sleep 60; rmdir $L ) &
  .venv/bin/python piper_sim/eval_piper.py --task-a $a --task-b $b --episodes 1 --start-episode $e $EVQ $BEST $FIX --out outputs/piper_fix_test/e$e 2>&1 \
    | grep -E "SUMMARY|Traceback|Error:" | grep -v EGL | sed "s/^/A$a B$b $e /" | cut -c1-260
  wait
done
echo 끝
