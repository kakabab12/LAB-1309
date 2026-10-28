#!/usr/bin/env bash
# DAgger 4차 (10/27 17:50): w5 (가까이 보기) 가 80~450스텝 하다 멈춘 자리에서 시범 프로그램이 이어받기.
#   w4 는 쥐기 80% 인데 놓기가 판 가운데에서 몇 cm 빗나감 → 이어받는 때를 d3 (40~450) 보다 늦게 (들고 가 놓는 단계 쪽으로).
#   w5 평가 (GPU 3줄) 와 같이 돌도록 1줄로 설정을 차례로. 끝나면 wfeat2 (사진) → wfeat3 (2판에서) → 사진 특징.
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
C=outputs/piper_w5_model/merged
log "[d4] w5 모델 기다림"
until [ -f $C/worldfeat_v3 ] && grep -q "\[③ w5\]" outputs/piper/world5.log; do
  systemctl --user is-active --quiet piper-w5 || { log "piper-w5 멈춤 — d4 안 함"; exit 1; }
  sleep 60
done
i=0
for c in default random many; do
  a=$((9750 + i * 200)); b=$((a + 199)); i=$((i + 1))
  log "[d4] $c $a-$b"
  PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_dagger.py --policy $C --seeds $a-$b --t 80 450 --out data/world_dagger4_$c $EVQ > outputs/piper/world_dagger4_$c.log 2>&1
  grep -h STATS outputs/piper/world_dagger4_$c.log
done
log "[d4] wfeat2 → wfeat3"
for c in default random many; do
  for s in 0 1 2; do ( PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_feat_cache.py --data data/world_dagger4_$c --shard $s --nshard 3 --stride 6 --version 2 | tail -1 ) & done; wait
done
D4="data/world_dagger4_default data/world_dagger4_random data/world_dagger4_many"
PIPER_WORLD=piper_sim/worlds/default.json $PY piper_sim/world_feat_cache.py --data $D4 --version 3 --from-v2 | tail -1
log "[d4] 사진 특징"
while systemctl --user is-active --quiet piper-w5; do sleep 60; done       # w5 평가와 GPU 겹치지 않게
for s in 0 1; do
  ( $PY vis_cache.py build --policy outputs/piper_np3_model/merged --data $D4 --out cache/vis --batch 8 --stride 6 --shard $s --nshard 2 2>&1 | grep -E "끝|Error|Traceback" ) &
done
wait
for d in $D4; do echo "$d npz $(ls $d/episodes/*.npz | wc -l) wfeat3 $(ls $d/episodes/*.wfeat3.npy | wc -l)"; done
log "끝"
