#!/usr/bin/env bash
# 고친 보조 나눠 보기 (10/23 새벽): VLA 손을 직접 잡는 두 가지(⑩ 쥐기 확정, ⑮ 먼저 들기)를 빼고
#   넓은 변환식 + ⑯ 판 따라가기 + ⑰ 기억 고치기 + ⑮' 먼 빈 자리만. A8→B0 20장면을 예전 보조(piper_np2mc)·고친 보조(piper_np2fc)와 짝 비교
cd "$(dirname "$0")"
export PIPER_NOPED=1 PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl PIPER_FRONT_CALIB=piper_sim/front_calib_wide.json
L=outputs/piper/gpu_launch.lock
BEST="--approach-assist --descend-assist --unjam-assist --yaw-assist --putdown-assist --fresh-held --putdown-near --wrong-guard --release-wait 40"
FIXB="--pad-track --mem-refresh --putdown-far"
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph --switch-at grasp:30 --strategy keep --policy outputs/piper_np2_model/merged"
for it in 8:0 8:7; do
  until mkdir $L 2>/dev/null; do sleep 10; done
  ( sleep 120; rmdir $L ) &
  .venv/bin/python piper_sim/eval_piper.py --task-a ${it%:*} --task-b ${it#*:} --episodes 20 --start-episode 1000 $EVQ $BEST $FIXB --out outputs/piper_np2fb_switch 2>&1 \
    | grep -E "SUMMARY|Traceback|Error:" | grep -v EGL | cut -c1-300
  wait
done
echo 끝
