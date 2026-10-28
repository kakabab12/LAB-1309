#!/usr/bin/env bash
# w7 (10/28 12시): DAgger 5차 (w6, 80~450스텝 뒤 이어받기, 세 설정 × 200장면, GPU 3줄) → wfeat2s (SURE) → wfeat3s → 사진 특징
#   → w7 학습 (w6 에서 15000, + d5) → 일 1개 혼자·다듬기 50장면 → 표 (w6, w6r, w7, w7r)
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
C6=outputs/piper_w6_model/merged
log "[① DAgger d5] w6, 세 설정 × 200장면 (10350~10949)"
i=0
for c in default random many; do
  a=$((10350 + i * 200)); b=$((a + 199)); i=$((i + 1))
  ( PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_dagger.py --policy $C6 --seeds $a-$b --t 80 450 --out data/world_dagger5_$c $EVQ > outputs/piper/world_dagger5_$c.log 2>&1 ) &
  sleep 20
done
wait
grep -h STATS outputs/piper/world_dagger5_*.log
wf2s(){ for s in 0 1 2; do ( PIPER_WORLD=piper_sim/worlds/$2.json nice -n 10 $PY piper_sim/world_feat_cache.py --data $1 --shard $s --nshard 3 --stride 6 --version 2 --sure 2>&1 | grep -E "^끝|Traceback|Error" ) & done; wait; }
D5="data/world_dagger5_default data/world_dagger5_random data/world_dagger5_many"
log "[② d5 wfeat2s·wfeat3s + 사진 특징]"
for s in 0 1; do
  ( $PY vis_cache.py build --policy outputs/piper_np3_model/merged --data $D5 --out cache/vis --batch 8 --stride 6 --shard $s --nshard 2 2>&1 | grep -E "끝|Error|Traceback" ) &
done
for c in default random many; do wf2s data/world_dagger5_$c $c; done
wait
DATA="data/world_default_t data/world_random data/world_many data/world_dagger_d1 data/world_dagger_r1 data/world_dagger2_default data/world_dagger2_random data/world_dagger2_many data/world_dagger3_default data/world_dagger3_random data/world_dagger3_many data/world_dagger4_default data/world_dagger4_random data/world_dagger4_many $D5"
PIPER_WORLD=piper_sim/worlds/default.json $PY piper_sim/world_feat_cache.py --data $DATA --version 3 --from-v2 --sure | tail -1
for d in $DATA; do
  a=$(ls $d/episodes/*.wfeat3s.npy | wc -l); b=$(ls $d/episodes/*.npz | wc -l)
  [ "$a" -eq "$b" ] || { log "$d wfeat3s $a / $b 모자람 — 멈춤"; exit 1; }
done
log "[③ w7 학습] $C6 에서 15000스텝, + d5 (MemAvailable $(grep MemAvailable /proc/meminfo | awk '{print int($2/1e6)}')GB)"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --log-every 500 --abs-pos --objfeat --feat wfeat3s --renorm-new-feat --renorm-action --vis-cache cache/vis --frame-stride 6 --sufstate --fp32-master --vlm-lora 16 --vlm-lora-lr 2e-4 --proj-lr 3e-4 --new-lr 3e-4"
$PY train_lora.py --policy $C6 --data $DATA $TR --batch-size 8 --grad-accum 2 --steps 15000 --lr 3e-5 --eval-every 5000 --save-every 5000 \
    --out outputs/piper_w7_model 2>&1 | grep --line-buffered -E '에피소드 학습|새로 쓰는|"step": [0-9]*000, "loss"|val_loss|모델 저장|Traceback|Error|Killed'
C=outputs/piper_w7_model/merged
[ -f $C/worldfeat_sure ] && [ -f $C/worldfeat_v3 ] || { log "w7 모델 표시 없음 — 멈춤"; exit 1; }
ev(){ ( PIPER_WORLD=piper_sim/worlds/$1.json $PY piper_sim/world_eval.py --policy $2 --seeds $3 --k 1 --nested $EVQ $5 --out $4/k1n.json 2>&1 | grep -E "SUMMARY|Traceback|Error" | sed "s|^|$4 |" ) & sleep 60; }
log "[④ w7] 일 1개 단독 (50장면, 세 설정)"
for c in default random many; do ev $c $C 1000-1049 outputs/world_w7_$c ""; done
wait
log "[④ w7] 일 1개 다듬기 (50장면, 세 설정)"
for c in default random many; do ev $c $C 1000-1049 outputs/world_w7r_$c "--refine grasp,place"; done
wait
for c in default random many; do PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_table.py outputs/world_w6_$c outputs/world_w6r_$c outputs/world_w7_$c outputs/world_w7r_$c; done
log "끝"
