#!/usr/bin/env bash
# 막힌 쥐기 마무리(⑲ --stuck-grasp 40) 시험 (10/23 03:35, ⑯ 뺀 고침 + ⑲) 로 노란 막대 B 두 쌍(A8→B7, A1→B7) 20장면씩.
#   비교: ⑯ 뺀 고침 (outputs/piper_np2fd_switch, run_np2h_round.sh), 예전 보조 (outputs/piper_np2mc_switch)
cd "$(dirname "$0")"
export PIPER_NOPED=1 PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl PIPER_FRONT_CALIB=piper_sim/front_calib_wide.json
L=outputs/piper/gpu_launch.lock
FIXB="--approach-assist --descend-assist --unjam-assist --yaw-assist --putdown-assist --fresh-held --putdown-near --wrong-guard --release-wait 40 --mem-refresh --putdown-far"
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph --switch-at grasp:30 --strategy keep --policy outputs/piper_np2_model/merged"
for it in 8:7 1:7; do
  until mkdir $L 2>/dev/null; do sleep 10; done
  ( sleep 120; rmdir $L ) &
  .venv/bin/python piper_sim/eval_piper.py --task-a ${it%:*} --task-b ${it#*:} --episodes 20 --start-episode 1000 $EVQ $FIXB --stuck-grasp 40 --out outputs/piper_np2sgd_switch 2>&1 \
    | grep -E "SUMMARY|Traceback|Error:" | grep -v EGL | cut -c1-300
  wait
done
echo 끝
