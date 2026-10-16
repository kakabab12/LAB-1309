#!/usr/bin/env bash
# 일지 GIF 용 영상 (10/17): als1 혼자 vs 정렬·놓기 보조, 같은 장면. 학습 옆에서 하나씩 (GPU 2.2GB)
set -u
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl
PY=.venv/bin/python; B=outputs/piper_als1_model/merged
Q="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph --video --episodes 1"
until mkdir outputs/piper/gpu_launch.lock 2>/dev/null; do sleep 15; done
$PY piper_sim/eval_piper.py --policy $B --task-a 0 --strategy none --start-episode 1000 $Q --out outputs/gif_als1_T0 2>&1 | grep -E "SUMMARY|Traceback|Error"
$PY piper_sim/eval_piper.py --policy $B --task-a 0 --strategy none --start-episode 1000 $Q --place-assist --out outputs/gif_als1place_T0 2>&1 | grep -E "SUMMARY|Traceback|Error"
$PY piper_sim/eval_piper.py --policy $B --task-a 8 --task-b 5 --switch-at grasp:3 --strategy keep --start-episode 1014 $Q --out outputs/gif_als1_A8B5 2>&1 | grep -E "SUMMARY|Traceback|Error"
$PY piper_sim/eval_piper.py --policy $B --task-a 8 --task-b 5 --switch-at grasp:3 --strategy keep --start-episode 1014 $Q --place-assist --out outputs/gif_als1place_A8B5 2>&1 | grep -E "SUMMARY|Traceback|Error"
rmdir outputs/piper/gpu_launch.lock
echo "=== $(date +%m/%d\ %H:%M) 끝"
