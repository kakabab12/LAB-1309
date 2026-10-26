#!/usr/bin/env bash
# w4 (10/27 02시 준비): ① piper-w3 끝나기 기다림 ② 고친 다듬기(두 번째 찾기가 로봇 가리개에 붙은 덩어리를 안 고름)로 w3 + 다듬기 일 1개 다시
#   ③ DAgger d3: w3 (wfeat2), 40~450스텝 뒤 이어받기, 세 설정 × 250장면 (9000~9749) ④ d3 wfeat2·사진 특징
#   ⑤ w4 학습 (w3 에서 15000) ⑥ 일 1개 단독·다듬기, 장면 1000~1049 (50장면, 논문용 구간 좁히기)
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
log "[w4] piper-w3 끝나기 기다림"
while systemctl --user is-active --quiet piper-w3; do sleep 60; done
C3=outputs/piper_w3_model/merged
ev(){ ( PIPER_WORLD=piper_sim/worlds/$1.json $PY piper_sim/world_eval.py --policy $2 --seeds $3 --k 1 --nested $EVQ $5 --out $4/k1n.json 2>&1 | grep -E "SUMMARY|Traceback|Error" | sed "s|^|$4 |" ) & sleep 60; }
log "[② w3 + 고친 다듬기] 일 1개 (세 설정)"
for c in default random many; do ev $c $C3 1000-1019 outputs/world_w3r2_$c "--refine grasp,place"; done
wait
for c in default random many; do PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_table.py outputs/world_w3_$c outputs/world_w3r_$c outputs/world_w3r2_$c; done
log "[③ DAgger d3] w3, 세 설정 × 250장면"
i=0
for c in default random many; do
  a=$((9000 + i * 250)); b=$((a + 249)); i=$((i + 1))
  ( PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_dagger.py --policy $C3 --seeds $a-$b --t 40 450 --out data/world_dagger3_$c $EVQ > outputs/piper/world_dagger3_$c.log 2>&1 ) &
  sleep 20
done
wait
grep -h STATS outputs/piper/world_dagger3_*.log
wf2(){ for s in 0 1 2 3; do ( PIPER_WORLD=piper_sim/worlds/$2.json $PY piper_sim/world_feat_cache.py --data $1 --shard $s --nshard 4 --stride 6 --version 2 | tail -1 ) & done; wait; }
D3="data/world_dagger3_default data/world_dagger3_random data/world_dagger3_many"
log "[④ d3 wfeat2 + 사진 특징]"
for s in 0 1; do
  ( $PY vis_cache.py build --policy outputs/piper_np3_model/merged --data $D3 --out cache/vis --batch 8 --stride 6 --shard $s --nshard 2 2>&1 | grep -E "끝|Error|Traceback" ) &
done
for c in default random many; do wf2 data/world_dagger3_$c $c; done
wait
DATA="data/world_default_t data/world_random data/world_many data/world_dagger_d1 data/world_dagger_r1 data/world_dagger2_default data/world_dagger2_random data/world_dagger2_many $D3"
for d in $DATA; do
  a=$(ls $d/episodes/*.wfeat2.npy | wc -l); b=$(ls $d/episodes/*.npz | wc -l)
  [ "$a" -eq "$b" ] || { log "$d wfeat2 $a / $b 모자람 — 멈춤"; exit 1; }
done
log "[⑤ w4 학습] $C3 에서 15000스텝 (MemAvailable $(grep MemAvailable /proc/meminfo | awk '{print int($2/1e6)}')GB)"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --log-every 500 --abs-pos --objfeat --feat wfeat2 --renorm-action --vis-cache cache/vis --frame-stride 6 --sufstate --fp32-master --vlm-lora 16 --vlm-lora-lr 2e-4 --proj-lr 3e-4 --new-lr 3e-4"
$PY train_lora.py --policy $C3 --data $DATA $TR --batch-size 8 --grad-accum 2 --steps 15000 --lr 3e-5 --eval-every 5000 --save-every 5000 \
    --out outputs/piper_w4_model 2>&1 | grep --line-buffered -E '에피소드 학습|"step": [0-9]*000, "loss"|val_loss|모델 저장|Traceback|Error|Killed'
C=outputs/piper_w4_model/merged
[ -f $C/config.json ] && [ -f $C/worldfeat_v2 ] || { log "학습 실패 — 멈춤"; exit 1; }
log "[⑥ w4] 일 1개 단독 (50장면, 세 설정)"
for c in default random many; do ev $c $C 1000-1049 outputs/world_w4_$c ""; done
wait
log "[⑥ w4] 일 1개 다듬기 (50장면, 세 설정)"
for c in default random many; do ev $c $C 1000-1049 outputs/world_w4r_$c "--refine grasp,place"; done
wait
for c in default random many; do PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_table.py outputs/world_w3_$c outputs/world_w3r2_$c outputs/world_w4_$c outputs/world_w4r_$c; done
log "끝"
