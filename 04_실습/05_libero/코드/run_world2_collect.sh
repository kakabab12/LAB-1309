#!/usr/bin/env bash
# 여러 모양·N개 장면 2라운드 시범 (10/24 저녁, 사용자 "모든 변수 환경등에서도 정확도 95% 이상"):
#   piper-np5 가 끝나 CPU·GPU 가 비면 (np5v 단독 혼자 10 뒤 멈춤) 3줄 더: 흔든 환경 random.json (장면 5000~5599, 일 1~4)
#   ‖ 물체 많고 색 겹침 many.json (장면 6000~6599, 일 1~6) — default 수집(piper-world)과 같이 돈다. w2 학습 재료.
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
log "[world2] piper-np5 끝나기 기다림"
while systemctl --user is-active --quiet piper-np5; do sleep 60; done
log "[world2 시범 수집] random 2줄 (5000~5599) + many 1줄 (6000~6599)"
( PIPER_WORLD=piper_sim/worlds/random.json $PY piper_sim/world_collect.py --seeds 5000-5299 --k 1 4 --nested-frac 0.7 --dart 0.1 --out data/world_random > outputs/piper/world2_collect_r0.log 2>&1 ) &
sleep 20
( PIPER_WORLD=piper_sim/worlds/random.json $PY piper_sim/world_collect.py --seeds 5300-5599 --k 1 4 --nested-frac 0.7 --dart 0.1 --out data/world_random > outputs/piper/world2_collect_r1.log 2>&1 ) &
sleep 20
( PIPER_WORLD=piper_sim/worlds/many.json $PY piper_sim/world_collect.py --seeds 6000-6599 --k 1 6 --nested-frac 0.7 --dart 0.1 --out data/world_many > outputs/piper/world2_collect_m0.log 2>&1 ) &
wait
grep -h STATS outputs/piper/world2_collect_*.log
log "끝: random $(ls data/world_random/episodes/*.npz | wc -l) 구간, many $(ls data/world_many/episodes/*.npz | wc -l) 구간"
