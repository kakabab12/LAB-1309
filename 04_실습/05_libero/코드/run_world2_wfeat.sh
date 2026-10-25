#!/usr/bin/env bash
# (10/25 14:50 다시) 물체 찾기 고친 뒤 위치 입력(wfeat) 다시 — 학습·평가가 같은 찾기를 쓰게. --stride 6: 학습(--frame-stride 6)이 읽는 장만.
#   w1 이 0% (예전 찾기로 만든 wfeat 는 지시문 물체를 15~31% 장면에서 아예 못 찾음) → default 도 바로: random 2 + many 2 + default 4 줄 (nice 10)
#   예전 default wfeat 는 data/world_default_t_oldwfeat
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
log "[wfeat] random 2 + many 2 + default 4"
for s in 0 1; do
  ( PIPER_WORLD=piper_sim/worlds/random.json $PY piper_sim/world_feat_cache.py --data data/world_random --shard $s --nshard 2 --stride 6 | tail -1 ) &
  ( PIPER_WORLD=piper_sim/worlds/many.json $PY piper_sim/world_feat_cache.py --data data/world_many --shard $s --nshard 2 --stride 6 | tail -1 ) &
done
for s in 0 1 2 3; do ( PIPER_WORLD=piper_sim/worlds/default.json $PY piper_sim/world_feat_cache.py --data data/world_default_t --shard $s --nshard 4 --stride 6 | tail -1 ) & done
wait
for d in data/world_random data/world_many data/world_default_t; do log "$d wfeat $(ls $d/episodes/*.wfeat.npy | wc -l) / $(ls $d/episodes/*.npz | wc -l)"; done
log "끝"
