#!/usr/bin/env bash
# w3 (10/26 10시 준비): w2 는 일 1개 단독 5% — 거의 다 물체까지 가고 3/4 은 쥐지만 놓기가 부정확 (판 가운데에서 4~17cm).
#   ① piper-w2eval 끝나기 기다림 ② DAgger d2: w2 가 40~450스텝 한 뒤 시범 프로그램이 이어받음 (들고 가 놓는 단계 바로잡기까지), 세 설정 1줄씩
#   ③ d2 wfeat·사진 특징 ④ w3 학습 (w2 에서 15000스텝, 모든 데이터 + d1·r1·d2) ⑤ 일 1·2개 단독 + 일 1개 보조, 세 설정
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
log "[w3] piper-w2eval 끝나기 기다림"
while systemctl --user is-active --quiet piper-w2eval; do sleep 120; done
P=outputs/piper_w2_model/merged
log "[DAgger d2] w2, 세 설정 × 250장면"
i=0
for c in default random many; do
  a=$((8000 + i * 250)); b=$((a + 249)); i=$((i + 1))
  ( PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_dagger.py --policy $P --seeds $a-$b --t 40 450 --out data/world_dagger2_$c $EVQ > outputs/piper/world_dagger2_$c.log 2>&1 ) &
  sleep 30
done
wait
grep -h STATS outputs/piper/world_dagger2_*.log
D2=""
for c in default random many; do
  d=data/world_dagger2_$c
  [ -d $d/episodes ] || continue
  for s in 0 1 2 3; do ( PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_feat_cache.py --data $d --shard $s --nshard 4 --stride 6 | tail -1 ) & done
  wait
  D2="$D2 $d"
done
log "[d2 사진 특징]$D2 ($(for d in $D2; do ls $d/episodes/*.npz; done | wc -l) 구간)"
for s in 0 1; do
  ( $PY vis_cache.py build --policy outputs/piper_np3_model/merged --data $D2 --out cache/vis --batch 8 --stride 6 --shard $s --nshard 2 2>&1 | grep -E "끝|Error|Traceback" ) &
done
wait
DATA="data/world_default_t data/world_random data/world_many data/world_dagger_d1 data/world_dagger_r1$D2"
for d in $DATA; do
  a=$(ls $d/episodes/*.wfeat.npy | wc -l); b=$(ls $d/episodes/*.npz | wc -l)
  [ "$a" -eq "$b" ] || { log "$d wfeat $a / $b 모자람 — 멈춤"; exit 1; }
done
log "[w3 학습] $P 에서 15000스텝 (MemAvailable $(grep MemAvailable /proc/meminfo | awk '{print int($2/1e6)}')GB)"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --log-every 500 --abs-pos --objfeat --feat wfeat --renorm-action --vis-cache cache/vis --frame-stride 6 --sufstate --fp32-master --vlm-lora 16 --vlm-lora-lr 2e-4 --proj-lr 3e-4 --new-lr 3e-4"
$PY train_lora.py --policy $P --data $DATA $TR --batch-size 8 --grad-accum 2 --steps 15000 --lr 3e-5 --eval-every 5000 --save-every 5000 \
    --out outputs/piper_w3_model 2>&1 | grep --line-buffered -E '에피소드 학습|"step": [0-9]*000, "loss"|val_loss|모델 저장|Traceback|Error|Killed'
C=outputs/piper_w3_model/merged
[ -f $C/config.json ] || { log "학습 실패 — 멈춤"; exit 1; }
FIX="--approach-assist --descend-assist --unjam-assist --yaw-assist --putdown-assist --fresh-held --putdown-near --wrong-guard --mem-refresh --putdown-far --stuck-grasp 40 --yaw-none-reset 10 --mem-tol 0.012 --open-empty --stack-guard --stack-clear --guard-r 0.07 --deliver"
log "[w3] 일 1·2개 단독 (세 설정, 6줄을 3·3)"
for k in 1 2; do
  for c in default random many; do
    ( PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_eval.py --policy $C --seeds 1000-1019 --k $k --nested $EVQ --out outputs/world_w3_$c/k${k}n.json 2>&1 | grep -E "SUMMARY|Traceback|Error" ) &
    sleep 60
  done
  wait
done
log "[w3 + 보조] 일 1개 (세 설정)"
for c in default random many; do
  ( PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_eval.py --policy $C --seeds 1000-1019 --k 1 --nested $EVQ $FIX --out outputs/world_w3a_$c/k1n.json 2>&1 | grep -E "SUMMARY|Traceback|Error" ) &
  sleep 60
done
wait
for c in default random many; do $PY piper_sim/world_fail.py outputs/world_w3_$c outputs/world_w3a_$c; done
log "끝"
