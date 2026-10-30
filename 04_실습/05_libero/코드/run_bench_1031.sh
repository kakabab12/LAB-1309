#!/usr/bin/env bash
# 10/31: 다듬기 쥐기 시험대 (CPU, OSMesa — GPU 안 씀): 예전 다듬기(RA) vs 새 다듬기(RB = + stk,q2,gq), 기본 설정 모양별 6장면 ×
#   위 카메라 치우침 0 / 1.5 / 2.5cm (네 방향). 두 줄로 나란히. 끄기는 systemctl --user stop piper-bench1031.
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PIPER_WORLD=piper_sim/worlds/default.json
PY=.venv/bin/python
RA="grasp,place,fmask,settle,wxy,gplan,objc,pfine"
S=1000,1001,1002,1003,1016,1026
nice -n 5 $PY piper_sim/refine_bench.py --seeds $S --parts $RA --err 0 1.5 2.5 --out outputs/piper/bench_ra.json > outputs/piper/bench_ra.log 2>&1 &
nice -n 5 $PY piper_sim/refine_bench.py --seeds $S --parts $RA,stk,q2,gq --err 0 1.5 2.5 --out outputs/piper/bench_rb.json > outputs/piper/bench_rb.log 2>&1 &
wait
grep -h SUMMARY outputs/piper/bench_ra.log outputs/piper/bench_rb.log
echo "=== $(date +%m/%d\ %H:%M) 끝"
