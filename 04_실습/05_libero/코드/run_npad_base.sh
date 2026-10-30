#!/usr/bin/env bash
# 10/30 낮: 사용자 '판도 n개로' — 평가가 판을 일 개수만큼(판 2개)만 놓던 것을 설정 n_pads 범위(일 개수 이상)로 고쳤다 (world.pick_n_pads).
#   판 N개에서 w9 혼자 일 1개 50장면 (세 설정) ‖ 멈춘 w10 파이프라인의 깊이 3 DAgger 기본 (남은 장면만, 판도 설정에서).
#   결과는 world_w9P_<설정> (P = 판 N개). 예전 판 2개 숫자(world_w9_<설정>)와 섞지 않는다. 끄기는 systemctl --user stop piper-npad.
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
C9=outputs/piper_w9_model/merged
P=()
log "[판 N개] w9 혼자 일 1개 50장면 ‖ 깊이 3 DAgger 기본 (남은 장면)"
( PIPER_WORLD=piper_sim/worlds/default.json $PY piper_sim/world_dagger_deep.py --policy $C9 --depth 3 --seeds 15000-15039 --out data/world_daggerK3_default $EVQ > outputs/piper/world_daggerK3_default_b.log 2>&1 ) & P+=($!); sleep 60
for c in default random many; do
  ( PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_eval.py --policy $C9 --seeds 1000-1049 --k 1 --nested $EVQ --out outputs/world_w9P_$c/k1n.json > outputs/piper/npad_w9_$c.log 2>&1 ) & P+=($!); sleep 60
done
wait "${P[@]}"
grep -h SUMMARY outputs/piper/npad_w9_*.log
grep -h STATS outputs/piper/world_daggerK3_default_b.log
log "끝"
