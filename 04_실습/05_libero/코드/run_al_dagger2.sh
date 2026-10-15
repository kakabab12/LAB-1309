#!/usr/bin/env bash
# 10/14 밤: run_al_dagger.sh 의 일 바꾸기·돌아오기 교정 시범이 GPU 메모리 부족으로 꺼져(같은 시간에 돌린 속도 시험 탓) 다시 모은다
set -u
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
D="--policy outputs/piper_bo1_model/merged --latency-steps 9 --ttrtc --n-action-steps 1 --strategy keep --cuda-graph --out data/piper_al_dagger_bo1"
log "일 바꾸기·돌아오기 교정 시범 시작"
( $PY piper_sim/dagger_piper.py $D --pairs 8:0,8:3,8:5,8:7,8:9,4:5 --episodes 3400-3409 --seed 125 > outputs/piper/al_dagger_5.log 2>&1 ) & P1=$!
( $PY piper_sim/dagger_piper.py $D --pairs 4:9,1:7,8:1,8:4,1:8,4:1 --episodes 3400-3409 --seed 126 > outputs/piper/al_dagger_6.log 2>&1 ) & P2=$!
( $PY piper_sim/dagger_piper.py $D --pairs 8:5,8:7,8:9,4:5,4:9,1:7 --resume --episodes 3400-3409 --seed 127 > outputs/piper/al_dagger_7.log 2>&1 ) & P3=$!
wait $P1 $P2 $P3
rm -rf data/piper_al_dagger_bo1_t
$PY piper_sim/trim_stalls.py data/piper_al_dagger_bo1 data/piper_al_dagger_bo1_t
$PY objfeat_cache.py --data data/piper_al_dagger_bo1_t
log "교정 시범2 끝: $(ls data/piper_al_dagger_bo1/episodes | wc -l)"
