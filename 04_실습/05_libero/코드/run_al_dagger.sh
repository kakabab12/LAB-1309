#!/usr/bin/env bash
# 10/14 bo1 교정 시범을 새 쥐기 방식(align)으로: 학생이 블록을 손목 화면 가운데로 못 맞춘 채 내려가면
#   시범 프로그램이 넘겨받아 위로 올라가 가운데로 맞추고 똑바로 내려가는 장면이 녹화된다 (영상 진단 10/14)
set -u
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
D="--policy outputs/piper_bo1_model/merged --latency-steps 9 --ttrtc --n-action-steps 1 --strategy keep --cuda-graph --out data/piper_al_dagger_bo1"
PAIRS=8:0,8:3,8:5,8:7,8:9,4:5,4:9,1:7,8:1,8:4,1:8,4:1
log "교정 시범 시작"
( $PY piper_sim/dagger_piper.py $D --tasks 0 1 2 3 4 --episodes 3400-3424 --seed 121 > outputs/piper/al_dagger_1.log 2>&1 ) & P1=$!
( $PY piper_sim/dagger_piper.py $D --tasks 5 6 7 8 9 --episodes 3400-3424 --seed 122 > outputs/piper/al_dagger_2.log 2>&1 ) & P2=$!
( $PY piper_sim/dagger_piper.py $D --pairs $PAIRS --episodes 3400-3409 --seed 123 > outputs/piper/al_dagger_3.log 2>&1
  $PY piper_sim/dagger_piper.py $D --pairs 8:5,8:7,8:9,4:5,4:9,1:7 --resume --episodes 3400-3409 --seed 124 > outputs/piper/al_dagger_4.log 2>&1 ) & P3=$!
wait $P1 $P2 $P3
$PY piper_sim/trim_stalls.py data/piper_al_dagger_bo1 data/piper_al_dagger_bo1_t
$PY objfeat_cache.py --data data/piper_al_dagger_bo1_t
log "교정 시범 끝: $(ls data/piper_al_dagger_bo1/episodes | wc -l)"
