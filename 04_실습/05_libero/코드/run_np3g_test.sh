#!/usr/bin/env bash
# 쌓은 블록 지키기 시험 (10/23 16:35): np3 + 새 설정(⑲⑨'⑰'⑳) + ㉒ --stack-guard + ㉑ --stack-clear, 쌓기 두 쌍 A8→B9·A4→B9 20장면씩.
#   비교: np3 + 새 설정 (outputs/piper_np3x_switch). 진단: outputs/piper/diag_stack3.log (B 무너짐 4장면 모두 든 손이 쌓은 블록 옆을 낮게 지나다 침)
cd "$(dirname "$0")"
export PIPER_NOPED=1 PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl PIPER_FRONT_CALIB=piper_sim/front_calib_wide.json
L=outputs/piper/gpu_launch.lock
FIX="--approach-assist --descend-assist --unjam-assist --yaw-assist --putdown-assist --fresh-held --putdown-near --wrong-guard --release-wait 40 --mem-refresh --putdown-far --stuck-grasp 40"
NEW="--yaw-none-reset 10 --mem-tol 0.012 --open-empty --stack-guard --stack-clear"
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph --switch-at grasp:30 --strategy keep --policy outputs/piper_np3_model/merged"
for it in 8:9 4:9; do
  until mkdir $L 2>/dev/null; do sleep 10; done
  ( sleep 120; rmdir $L ) &
  .venv/bin/python piper_sim/eval_piper.py --task-a ${it%:*} --task-b ${it#*:} --episodes 20 --start-episode 1000 $EVQ $FIX $NEW --out outputs/piper_np3g_switch 2>&1 \
    | grep -E "SUMMARY|Traceback|Error:" | grep -v EGL | cut -c1-300
  wait
done
echo 끝
