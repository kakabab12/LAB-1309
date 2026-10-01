#!/usr/bin/env bash
# 6차 이후 순서 (2026-10-01 20:30 결정)
#   4000스텝 미리 보기가 나빠서 (가운데 서랍 0/10, 그릇→접시 1/10, 접시 밀기 0/10: 큰 동작은 배웠지만
#   마지막 1~2cm 정밀 동작을 못 함) 덜 학습된 모델을 길게 평가하는 대신 학습을 더 한다.
# ⚠️ 메모리: 학습(18GB)과 다른 무거운 작업을 겹치지 않는다. 평가·DAgger 는 동시에 3개까지.
cd "$(dirname "$0")"
PY=.venv/bin/python
F='SUMMARY|STATS|Traceback|Error|  =='
log(){ echo "=== $(date +%H:%M:%S) $*"; }
training(){ ps -eo comm,args | awk '$1=="python" && /train_lora/ && !/awk/' | grep -q .; }
EV10="--episodes 10 --start-episode 20"
EV30="--episodes 30 --start-episode 20"
FAIL="8:0 8:3 8:5 8:7 8:9 4:5 4:9 1:7"
OK="8:1 8:4 1:8 4:1"
forget(){ local POL=$1 OUT=$2 EV=$3; shift 3; for t in "$@"; do $PY switch_experiment.py --policy $POL --task-a $t --strategy none $EV --out $OUT 2>&1 | grep -E "$F"; done; }
switch(){ local POL=$1 OUT=$2 EV=$3; shift 3; for p in "$@"; do $PY switch_experiment.py --policy $POL --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy flush $EV --out $OUT 2>&1 | grep -E "$F"; done; }

log "[1] 6차 학습(24000스텝)이 끝나기를 기다림"
while training; do sleep 60; done
[ -f outputs/v6_model/merged/model.safetensors ] || { log "6차 모델이 없다 — 멈춤"; exit 1; }

log "[2] 짧은 평가 (주요 항목 10회)"
( forget outputs/v6_model/merged outputs/v6q_forget "$EV10" 0 3 5 8 9 ) &
( switch outputs/v6_model/merged outputs/v6q_switch "$EV10" 8:7 8:3 8:5 8:1 ) &
wait

log "[3] 6b차 학습: 6차에서 이어서 36000스텝"
$PY train_lora.py --policy outputs/v6_model/merged --data data/v6_normal data/v6_switch --full-expert --aug --balance \
  --workers 0 --steps 36000 --batch-size 4 --grad-accum 2 --lr 3e-5 --eval-every 3000 --save-every 6000 --log-every 500 \
  --out outputs/v6b_model 2>&1 | grep --line-buffered -E '에피소드 학습|val_loss|모델 저장|Traceback|Error'

log "[4] 6b차 정식 평가 (장면 20~49, 30회)"
( forget outputs/v6b_model/merged outputs/v6eval_forget "$EV30" 0 1 2 3 4 5 6 7 8 9 ) &
( switch outputs/v6b_model/merged outputs/v6eval_switch "$EV30" $FAIL $OK ) &
wait

log "[5] DAgger 수집 (6b차) + 원래 모델 같은 장면 평가 (장면 30~49)"
( $PY dagger_v6.py --policy outputs/v6b_model/merged --tasks 0 1 2 3 4 5 6 7 8 9 \
    --pairs 8:0,8:3,8:5,8:7,8:9,4:5,4:9,1:7,8:1,8:4,1:8,4:1 --episodes 0-19,50-69 --dart 0.1 \
    --out data/v6_dagger1 2>&1 | grep -E "$F" ) &
( forget HuggingFaceVLA/smolvla_libero outputs/v6eval_forget_base "--episodes 20 --start-episode 30" 0 1 2 3 4 5 6 7 8 9 ) &
( switch HuggingFaceVLA/smolvla_libero outputs/v6eval_switch_base "--episodes 20 --start-episode 30" $FAIL $OK ) &
wait

log "[6] 6c차 학습: 6b차에서 이어서, DAgger 시범 더해 12000스텝"
$PY train_lora.py --policy outputs/v6b_model/merged --data data/v6_normal data/v6_switch data/v6_dagger1 --full-expert --aug \
  --balance --workers 0 --steps 12000 --batch-size 4 --grad-accum 2 --lr 2e-5 --eval-every 3000 --save-every 6000 \
  --log-every 500 --out outputs/v6c_model 2>&1 | grep --line-buffered -E '에피소드 학습|val_loss|모델 저장|Traceback|Error'

log "[7] 6c차 정식 평가 + 지연 조건 (0.56초)"
( forget outputs/v6c_model/merged outputs/v6ceval_forget "$EV30" 0 1 2 3 4 5 6 7 8 9 ) &
( switch outputs/v6c_model/merged outputs/v6ceval_switch "$EV30" $FAIL $OK ) &
wait
( forget outputs/v6c_model/merged outputs/v6lat_forget_v6c "$EV10 --latency-steps 11" 0 1 2 3 4 5 6 7 8 9 ) &
( switch outputs/v6c_model/merged outputs/v6lat_switch_v6c "$EV10 --latency-steps 11" $FAIL $OK ) &
( forget HuggingFaceVLA/smolvla_libero outputs/v6lat_forget_base "$EV10 --latency-steps 11" 0 1 2 3 4 5 6 7 8 9 ) &
wait
log "끝"
