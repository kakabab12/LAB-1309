#!/usr/bin/env bash
# run_alr_round.sh 이어 하기 (10/15 18:50): 교정 시범을 멈추다 순서 스크립트 본체까지 꺼져서, 돌고 있는 alr1 학습(PID 로 기다림) 뒤를 잇는다
set -u
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --log-every 500 --abs-pos --objfeat --feat relfeat --vis-cache cache/vis --frame-stride 3"
DATA="data/piper_al_normal_t data/piper_al_wide_normal_t data/piper_al_switch_t data/piper_al_recovery_t data/piper_al_dagger_bo1_t"
ev_t(){ local pol=$1 o=$2; shift 2; for t in "$@"; do $PY piper_sim/eval_piper.py --policy $pol --task-a $t --strategy none --episodes 20 --start-episode 1000 $EVQ --out outputs/${o}_forget 2>&1 | grep -E "$F"; done; }
ev_p(){ local pol=$1 o=$2; shift 2; for p in "$@"; do $PY piper_sim/eval_piper.py --policy $pol --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy keep --episodes 20 --start-episode 1000 $EVQ --out outputs/${o}_switch 2>&1 | grep -E "$F"; done; }
board(){ $PY scoreboard.py --blocks --forget outputs/$1_forget --switch outputs/$1_switch --name "$2" --png outputs/media/score_$1.png --md outputs/piper/score_$1.md | tail -1
         $PY scoreboard.py --blocks --loose --forget outputs/$1_forget --switch outputs/$1_switch --name "$2 판 안 기준" --png outputs/media/score_$1_loose.png --md outputs/piper/score_$1_loose.md | tail -1; }
all28(){ local pol=$1 o=$2
  ( ev_t $pol $o 0 1 2 3 4 ) & local P1=$!; ( ev_t $pol $o 5 6 7 8 9 ) & local P2=$!
  ( ev_p $pol $o 8:0 8:3 8:5 8:7 8:9 4:5 ) & local P3=$!; ( ev_p $pol $o 4:9 1:7 8:1 8:4 1:8 4:1 ) & local P4=$!
  wait $P1 $P2 $P3 $P4; }
prep_data(){ local D=$1 B=$2
  rm -rf ${D}_t; $PY piper_sim/trim_stalls.py $D ${D}_t
  for s in 0 1 2; do ( $PY objfeat_cache.py --data ${D}_t --shard $s --nshard 3 > /dev/null 2>&1 ) & done; wait
  $PY relfeat.py --data ${D}_t
  ( $PY vis_cache.py build --policy $B --data ${D}_t --out cache/vis --batch 8 --stride 3 --shard 0 --nshard 2 2>&1 | grep -E "끝|Error|Traceback" ) & local Q1=$!
  ( $PY vis_cache.py build --policy $B --data ${D}_t --out cache/vis --batch 8 --stride 3 --shard 1 --nshard 2 2>&1 | grep -E "끝|Error|Traceback" ) & local Q2=$!
  wait $Q1 $Q2; }

while kill -0 163341 2>/dev/null; do sleep 60; done
B=outputs/piper_alr1_model/merged
[ -f $B/relfeat ] || { log "alr1 모델 없음 — 멈춤"; exit 1; }
log "[2] alr1 28항목 (지연 9)"
all28 $B piper_alr1h
board piper_alr1h "손 기준 물체 위치 alr1 (새 배치 20장면, 지연 9)"
OUTD=data/piper_al_dagger_alr1
eval "$($PY piper_sim/plan_dagger.py --base piper_alr1h --start 3600)"
log "[3] alr1 교정 시범 2개 (CUDA 그래프 없이 — 남는 자리에서 보조 켬 평가가 같이 돈다) 단독 30:[${T30}] 20:[${T20}] / 전환 30:[${P30}] 20:[${P20}] / 재개 30:[${R30}] 20:[${R20}]"
DG="--policy $B --latency-steps 9 --ttrtc --n-action-steps 1 --strategy keep --max-policy 350 --out $OUTD"
dag_t(){ for k in 30 20 10; do v=T$k; e=E$k; [ -n "${!v}" ] && $PY piper_sim/dagger_piper.py $DG --tasks ${!v} --episodes ${!e} --seed $((141 + k)); done; }
dag_p(){ for k in 30 20 10; do v=P$k; e=E$k; [ -n "${!v}" ] && $PY piper_sim/dagger_piper.py $DG --pairs ${!v} --episodes ${!e} --seed $((142 + k)); done; }
dag_r(){ for k in 30 20 10; do v=R$k; e=E$k; [ -n "${!v}" ] && $PY piper_sim/dagger_piper.py $DG --pairs ${!v} --resume --episodes ${!e} --seed $((143 + k)); done; }
( dag_t > outputs/piper/dagger_alr1_t.log 2>&1; dag_r > outputs/piper/dagger_alr1_r.log 2>&1 ) & D1=$!
( dag_p > outputs/piper/dagger_alr1_p.log 2>&1 ) & D2=$!
wait $D1 $D2
log "교정 시범 끝: al1 $(ls data/piper_al_dagger_al1/episodes | wc -l), alr1 $(ls $OUTD/episodes | wc -l)"
prep_data data/piper_al_dagger_al1 $B
prep_data $OUTD $B
log "[4] alr2 학습 (alr1 에서, + al1·alr1 교정 시범)"
$PY train_lora.py --policy $B --data $DATA data/piper_al_dagger_al1_t ${OUTD}_t $TR \
  --dagger-frac 0.35 --batch-size 8 --grad-accum 2 --steps 10000 --lr 3e-5 --eval-every 2500 --save-every 5000 \
  --out outputs/piper_alr2_model 2>&1 | grep --line-buffered -E '에피소드 학습|val_loss|모델 저장|Traceback|Error'
B2=outputs/piper_alr2_model/merged
[ -f $B2/relfeat ] || { log "alr2 모델 없음 — 멈춤"; exit 1; }
log "[5] alr2 28항목"
all28 $B2 piper_alr2h
board piper_alr2h "손 기준 물체 위치 alr2 (새 배치 20장면, 지연 9)"
log "끝"
