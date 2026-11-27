#!/bin/bash
# placo 비교 (11/27): ① numpy 자료로 배운 정책을 placo 로 실행 ② placo 자료로 배우고 placo 로 실행. 평가 장면은 같은 50개 (seed 1000)
set -e
cd /home/user/smolVLA
PY=point_policy/.venv/bin/python
export MUJOCO_GL=egl
$PY point_policy/sim_eval.py point_policy/runs/synth40/policy.pt --ik placo --n 50 --out point_policy/runs/synth40/eval_seed1000_placo.json
$PY point_policy/train.py point_policy/data/cube_to_pad/synth/dataset_placo.npz --out point_policy/runs/synth40_placo --steps 20000
$PY point_policy/sim_eval.py point_policy/runs/synth40_placo/policy.pt --ik placo --n 50 --out point_policy/runs/synth40_placo/eval_seed1000.json
echo "=== $(date '+%F %T') 끝"
