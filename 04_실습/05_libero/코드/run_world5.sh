#!/usr/bin/env bash
# w5 (10/27 12시 준비): w4 는 놓을 판 위치 입력이 0.1~2cm 로 맞는데도 판 가운데에서 5~15cm 떨어진 곳에서 손을 폈다 (w4rel many).
#   위치 입력 3판 (world_feat.derive_v3: [11:15] tanh((물체·놓을 곳 − 손) / 판 크기 절반), 가까이 보기) 을 더해 w4 에서 15000스텝.
#   데이터는 w4 와 같음 (DAgger 새로 안 모음 — 3판의 효과만 보려고). *.wfeat3.npy 는 2판 파일에서 바로 (world_feat_cache --from-v2)
#   ① piper-w4 끝나기 기다림 ② w5 학습 ③ 일 1개 단독·다듬기, 장면 1000~1049 ④ 표 (w4, w4r, w5, w5r)
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
log "[w5] piper-w4 끝나기 기다림"
while systemctl --user is-active --quiet piper-w4; do sleep 60; done
C4=outputs/piper_w4_model/merged
DATA="data/world_default_t data/world_random data/world_many data/world_dagger_d1 data/world_dagger_r1 data/world_dagger2_default data/world_dagger2_random data/world_dagger2_many data/world_dagger3_default data/world_dagger3_random data/world_dagger3_many"
PIPER_WORLD=piper_sim/worlds/default.json $PY piper_sim/world_feat_cache.py --data $DATA --version 3 --from-v2 | tail -1
for d in $DATA; do
  a=$(ls $d/episodes/*.wfeat3.npy | wc -l); b=$(ls $d/episodes/*.npz | wc -l)
  [ "$a" -eq "$b" ] || { log "$d wfeat3 $a / $b 모자람 — 멈춤"; exit 1; }
done
log "[② w5 학습] $C4 에서 15000스텝, wfeat3 (MemAvailable $(grep MemAvailable /proc/meminfo | awk '{print int($2/1e6)}')GB)"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --log-every 500 --abs-pos --objfeat --feat wfeat3 --renorm-new-feat --renorm-action --vis-cache cache/vis --frame-stride 6 --sufstate --fp32-master --vlm-lora 16 --vlm-lora-lr 2e-4 --proj-lr 3e-4 --new-lr 3e-4"
$PY train_lora.py --policy $C4 --data $DATA $TR --batch-size 8 --grad-accum 2 --steps 15000 --lr 3e-5 --eval-every 5000 --save-every 5000 \
    --out outputs/piper_w5_model 2>&1 | grep --line-buffered -E '에피소드 학습|새로 쓰는|"step": [0-9]*000, "loss"|val_loss|모델 저장|Traceback|Error|Killed'
C=outputs/piper_w5_model/merged
[ -f $C/worldfeat_v3 ] || { log "w5 모델에 worldfeat_v3 표시 없음 — 멈춤"; exit 1; }
ev(){ ( PIPER_WORLD=piper_sim/worlds/$1.json $PY piper_sim/world_eval.py --policy $2 --seeds $3 --k 1 --nested $EVQ $5 --out $4/k1n.json 2>&1 | grep -E "SUMMARY|Traceback|Error" | sed "s|^|$4 |" ) & sleep 60; }
log "[③ w5] 일 1개 단독 (50장면, 세 설정)"
for c in default random many; do ev $c $C 1000-1049 outputs/world_w5_$c ""; done
wait
log "[③ w5] 일 1개 다듬기 (50장면, 세 설정)"
for c in default random many; do ev $c $C 1000-1049 outputs/world_w5r_$c "--refine grasp,place"; done
wait
for c in default random many; do PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_table.py outputs/world_w4_$c outputs/world_w4r_$c outputs/world_w5_$c outputs/world_w5r_$c; done
log "끝"
