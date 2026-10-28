#!/usr/bin/env bash
# w8 (10/28 18시): 일 2개 끼어들기가 약함 (w6 + 다듬기 default 둘 다 2/20, 넘어간 14장면 중 6장면은 B 를 못 쥠).
#   ① piper-w7 끝나기 기다림 ② 끼어들기 DAgger (w7, world_dagger_nested: A 를 30스텝 쥐면 B 지시, VLA 가 0~300스텝 하다 시범 프로그램이
#   A 내려놓기 → B → A 돌아오기; 세 설정 × 200장면 11000~11599) ‖ w7 + 다듬기 일 2개 default 20장면 (기준)
#   ③ wfeat2s → wfeat3s + 사진 특징 ④ w8 학습 (w7 에서 15000, + 끼어들기 DAgger) ⑤ w8 일 1개 혼자·다듬기 50장면, 일 2개 다듬기 20장면 (세 설정)
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
log "[w8] piper-w7 끝나기 기다림"
while systemctl --user is-active --quiet piper-w7; do sleep 60; done
C7=outputs/piper_w7_model/merged
[ -f $C7/worldfeat_sure ] || { log "w7 모델 없음 — 멈춤"; exit 1; }
ev(){ ( PIPER_WORLD=piper_sim/worlds/$1.json $PY piper_sim/world_eval.py --policy $2 --seeds $3 --k $6 --nested $EVQ $5 --out $4/k${6}n.json 2>&1 | grep -E "SUMMARY|Traceback|Error" | sed "s|^|$4 k$6 |" ) & sleep 60; }
log "[② 끼어들기 DAgger] w7, 세 설정 × 200장면 ‖ w7r 일 2개 default"
i=0
for c in default random many; do
  a=$((11000 + i * 200)); b=$((a + 199)); i=$((i + 1))
  ( PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_dagger_nested.py --policy $C7 --seeds $a-$b --out data/world_daggerN_$c $EVQ > outputs/piper/world_daggerN_$c.log 2>&1 ) &
  sleep 20
done
ev default $C7 1000-1019 outputs/world_w7r_default "--refine grasp,place" 2
wait
grep -h STATS outputs/piper/world_daggerN_*.log
DN="data/world_daggerN_default data/world_daggerN_random data/world_daggerN_many"
wf2s(){ for s in 0 1 2; do ( PIPER_WORLD=piper_sim/worlds/$2.json nice -n 10 $PY piper_sim/world_feat_cache.py --data $1 --shard $s --nshard 3 --stride 6 --version 2 --sure 2>&1 | grep -E "^끝|Traceback|Error" ) & done; wait; }
log "[③ 끼어들기 DAgger wfeat2s·wfeat3s + 사진 특징]"
for s in 0 1; do
  ( $PY vis_cache.py build --policy outputs/piper_np3_model/merged --data $DN --out cache/vis --batch 8 --stride 6 --shard $s --nshard 2 2>&1 | grep -E "끝|Error|Traceback" ) &
done
for c in default random many; do wf2s data/world_daggerN_$c $c; done
wait
D45="data/world_dagger4_default data/world_dagger4_random data/world_dagger4_many data/world_dagger5_default data/world_dagger5_random data/world_dagger5_many"
DATA="data/world_default_t data/world_random data/world_many data/world_dagger_d1 data/world_dagger_r1 data/world_dagger2_default data/world_dagger2_random data/world_dagger2_many data/world_dagger3_default data/world_dagger3_random data/world_dagger3_many $D45 $DN"
PIPER_WORLD=piper_sim/worlds/default.json $PY piper_sim/world_feat_cache.py --data $DATA --version 3 --from-v2 --sure | tail -1
for d in $DATA; do
  a=$(ls $d/episodes/*.wfeat3s.npy | wc -l); b=$(ls $d/episodes/*.npz | wc -l)
  [ "$a" -eq "$b" ] || { log "$d wfeat3s $a / $b 모자람 — 멈춤"; exit 1; }
done
log "[④ w8 학습] $C7 에서 15000스텝, + 끼어들기 DAgger (MemAvailable $(grep MemAvailable /proc/meminfo | awk '{print int($2/1e6)}')GB)"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --log-every 500 --abs-pos --objfeat --feat wfeat3s --renorm-new-feat --renorm-action --vis-cache cache/vis --frame-stride 6 --sufstate --fp32-master --vlm-lora 16 --vlm-lora-lr 2e-4 --proj-lr 3e-4 --new-lr 3e-4"
$PY train_lora.py --policy $C7 --data $DATA $TR --batch-size 8 --grad-accum 2 --steps 15000 --lr 3e-5 --eval-every 5000 --save-every 5000 \
    --out outputs/piper_w8_model 2>&1 | grep --line-buffered -E '에피소드 학습|새로 쓰는|"step": [0-9]*000, "loss"|val_loss|모델 저장|Traceback|Error|Killed'
C=outputs/piper_w8_model/merged
[ -f $C/worldfeat_sure ] && [ -f $C/worldfeat_v3 ] || { log "w8 모델 표시 없음 — 멈춤"; exit 1; }
log "[⑤ w8] 일 1개 혼자 (50장면, 세 설정)"
for c in default random many; do ev $c $C 1000-1049 outputs/world_w8_$c "" 1; done
wait
log "[⑤ w8] 일 1개 다듬기 (50장면) + 일 2개 다듬기 (20장면, default)"
for c in default random many; do ev $c $C 1000-1049 outputs/world_w8r_$c "--refine grasp,place" 1; done
wait
for c in default random many; do ev $c $C 1000-1019 outputs/world_w8r_$c "--refine grasp,place" 2; done
wait
for c in default random many; do PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_table.py outputs/world_w7_$c outputs/world_w7r_$c outputs/world_w8_$c outputs/world_w8r_$c; done
log "끝"
