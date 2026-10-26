#!/usr/bin/env bash
# w3 (10/26 16시 바꿈): 위치 입력 2판 (wfeat2 = 1판 + 옮길 물체 방향 [8:11]) 으로 학습 — w2 가 막대(방향을 맞춰야 쥠)를 거의 못 쥠.
#   ① DAgger d2 이어서 (w2, 40~450스텝, 세 설정 250장면씩; 있는 장면은 건너뜀) ‖ 기존 데이터 wfeat2 (CPU 4줄, nice)
#   ② d2 wfeat2 + 사진 특징 ③ w3 학습 (w2 에서 15000, --feat wfeat2 --renorm-new-feat) ④ 일 1·2개 단독 + 일 1개 다듬기, 세 설정
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
P=outputs/piper_w2_model/merged
log "[① DAgger d2 이어서 ‖ wfeat2 기존 데이터]"
i=0
for c in default random many; do
  a=$((8000 + i * 250)); b=$((a + 249)); i=$((i + 1))
  ( PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_dagger.py --policy $P --seeds $a-$b --t 40 450 --out data/world_dagger2_$c $EVQ >> outputs/piper/world_dagger2_$c.log 2>&1 ) &
  sleep 20
done
wf2(){ for s in 0 1 2 3; do ( PIPER_WORLD=piper_sim/worlds/$2.json nice -n 10 $PY piper_sim/world_feat_cache.py --data $1 --shard $s --nshard 4 --stride 6 --version 2 | tail -1 ) & done; wait; }
( wf2 data/world_default_t default; wf2 data/world_dagger_d1 default; wf2 data/world_random random; wf2 data/world_dagger_r1 random; wf2 data/world_many many ) &
wait
grep -h STATS outputs/piper/world_dagger2_*.log | tail -3
log "[② d2 wfeat2 + 사진 특징]"
D2="data/world_dagger2_default data/world_dagger2_random data/world_dagger2_many"
for s in 0 1; do
  ( $PY vis_cache.py build --policy outputs/piper_np3_model/merged --data $D2 --out cache/vis --batch 8 --stride 6 --shard $s --nshard 2 2>&1 | grep -E "끝|Error|Traceback" ) &
done
for c in default random many; do wf2 data/world_dagger2_$c $c; done
wait
DATA="data/world_default_t data/world_random data/world_many data/world_dagger_d1 data/world_dagger_r1 $D2"
for d in $DATA; do
  a=$(ls $d/episodes/*.wfeat2.npy | wc -l); b=$(ls $d/episodes/*.npz | wc -l)
  log "$d wfeat2 $a / $b"
  [ "$a" -eq "$b" ] || { log "wfeat2 모자람 — 멈춤"; exit 1; }
done
log "[③ w3 학습] $P 에서 15000스텝, wfeat2 (MemAvailable $(grep MemAvailable /proc/meminfo | awk '{print int($2/1e6)}')GB)"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --log-every 500 --abs-pos --objfeat --feat wfeat2 --renorm-new-feat --renorm-action --vis-cache cache/vis --frame-stride 6 --sufstate --fp32-master --vlm-lora 16 --vlm-lora-lr 2e-4 --proj-lr 3e-4 --new-lr 3e-4"
$PY train_lora.py --policy $P --data $DATA $TR --batch-size 8 --grad-accum 2 --steps 15000 --lr 3e-5 --eval-every 5000 --save-every 5000 \
    --out outputs/piper_w3_model 2>&1 | grep --line-buffered -E '에피소드 학습|새로 쓰는|"step": [0-9]*000, "loss"|val_loss|모델 저장|Traceback|Error|Killed'
C=outputs/piper_w3_model/merged
[ -f $C/config.json ] && [ -f $C/worldfeat_v2 ] || { log "학습 실패 (또는 worldfeat_v2 표시 없음) — 멈춤"; exit 1; }
ev(){ ( PIPER_WORLD=piper_sim/worlds/$1.json $PY piper_sim/world_eval.py --policy $C --seeds 1000-1019 --k $2 --nested $EVQ $4 --out $3/k$2n.json 2>&1 | grep -E "SUMMARY|Traceback|Error" | sed "s|^|$3 |" ) & sleep 60; }
log "[④ w3] 일 1개 단독 ‖ 다듬기 (세 설정)"
for c in default random many; do ev $c 1 outputs/world_w3_$c ""; done
wait
for c in default random many; do ev $c 1 outputs/world_w3r_$c "--refine grasp,place"; done
wait
for c in default random many; do ev $c 2 outputs/world_w3_$c ""; done
wait
for c in default random many; do PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_table.py outputs/world_w2_$c outputs/world_w2r_$c outputs/world_w3_$c outputs/world_w3r_$c; done
$PY piper_sim/world_fail.py outputs/world_w3_default outputs/world_w3r_default outputs/world_w3_random outputs/world_w3r_random outputs/world_w3_many outputs/world_w3r_many
log "끝"
