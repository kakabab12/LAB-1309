#!/bin/bash
# 가짜 시범 40개를 두 카메라 영상으로 그리고 → 첫 장면 점(직접 찍은 것처럼 참값+1.5px) → CoTracker3 → 학습 → 시뮬 평가 (11/27)
set -e
cd /home/user/smolVLA
PY=.venv/bin/python
export MUJOCO_GL=egl
D=point_policy/data/cube_to_pad/synth_video40
$PY point_policy/render_synth_video.py point_policy/data/cube_to_pad/synth --n 40 --out $D --click-all
$PY point_policy/object_points.py $D --redo
$PY point_policy/build_dataset.py $D
$PY point_policy/train.py $D/dataset.npz --out point_policy/runs/synth40_tracked --steps 20000
$PY point_policy/sim_eval.py point_policy/runs/synth40_tracked/policy.pt --n 50 --out point_policy/runs/synth40_tracked/eval_seed1000.json
echo "=== $(date '+%F %T') 끝"
