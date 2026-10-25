#!/usr/bin/env bash
# 여러 모양 장면 DAgger d1 (10/25 21시): w1b 가 무작위 20~300 스텝 한 뒤 시범 프로그램이 이어받아 마무리 (선생 구간만 저장).
#   default 1줄 (7000~7499) + random 1줄 (7500~7999). run_world2_train.sh 가 사진 특징이 끝나면 이 단위를 멈추고 모은 만큼 w2 에 넣는다.
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
( PIPER_WORLD=piper_sim/worlds/default.json $PY piper_sim/world_dagger.py --policy outputs/piper_w1b_model/merged --seeds 7000-7499 --out data/world_dagger_d1 $EVQ > outputs/piper/world_dagger_d0.log 2>&1 ) &
sleep 30
( PIPER_WORLD=piper_sim/worlds/random.json $PY piper_sim/world_dagger.py --policy outputs/piper_w1b_model/merged --seeds 7500-7999 --out data/world_dagger_r1 $EVQ > outputs/piper/world_dagger_r0.log 2>&1 ) &
wait
