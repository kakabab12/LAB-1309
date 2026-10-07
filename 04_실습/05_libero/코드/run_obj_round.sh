#!/usr/bin/env bash
# 위치 입력 라운드 (10/6 14:45 다시 걺 — 14시쯤 메모리가 바닥나 모든 프로세스가 죽었다. ba2 는 저장 전에 죽어 버림)
#   ① 검출 미리 계산이 끝나면 bo1 (ba1 에서, 절대 목표 + 물체 위치 24개, 사진 3장에 1장 — 메모리 약 9GB)
#   ② bo1 28항목 (3cm·판 안) ③ bo1 이 95% 못 넘는 항목 교정 시범 → 검출 계산 ④ bo2 → 28항목
set -u
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 MUJOCO_GL=egl
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 11 --ttrtc --n-action-steps 1"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --frame-stride 3 --batch-size 4 --grad-accum 2 --log-every 500"
DATA="data/piper_blk_dagger_b1_t data/piper_blk_dagger_ba1_t data/piper_blk_normal2_t data/piper_blk_normal3_t data/piper_blk_normal4_t data/piper_blk_normal_t data/piper_blk_recovery_t data/piper_blk_switch2_t data/piper_blk_switch3_t data/piper_blk_switch4_t data/piper_blk_switch_t data/piper_blk_wide_normal_t data/piper_blk_wide_recovery_t data/piper_blk_wide_switch_t "
ev_t(){ local pol=$1 o=$2 opt=$3; shift 3; for t in "$@"; do $PY piper_sim/eval_piper.py --policy $pol --task-a $t --strategy none --episodes 20 --start-episode 1000 $opt --out outputs/${o}_forget 2>&1 | grep -E "$F"; done; }
ev_p(){ local pol=$1 o=$2 opt=$3; shift 3; for p in "$@"; do $PY piper_sim/eval_piper.py --policy $pol --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy keep --episodes 20 --start-episode 1000 $opt --out outputs/${o}_switch 2>&1 | grep -E "$F"; done; }
board(){ $PY scoreboard.py --blocks --forget outputs/$1_forget --switch outputs/$1_switch --name "$2" --png outputs/media/score_$1.png --md outputs/piper/score_$1.md | tail -1
         $PY scoreboard.py --blocks --loose --forget outputs/$1_forget --switch outputs/$1_switch --name "$2 판 안 기준" --png outputs/media/score_$1_loose.png --md outputs/piper/score_$1_loose.md | tail -1; }
# 같은 PC 에서 다른 세션의 ACT 작업(train_act.py)이 돌면 평가를 2개만 띄운다 (10/6 14시 메모리 바닥 → 모든 프로세스 죽음)
other(){ ps -eo args | grep -cE "^\.venv/bin/python (train_act|gen_data)\.py"; }
all28(){ local pol=$1 o=$2
  if [ "$(other)" -gt 0 ]; then
    log "  다른 세션 ACT 작업 $(other)개가 돌아 평가 2개만"
    ( ev_t $pol $o "$EVQ" 0 1 2 3 4 5 6 7 8 9; ev_p $pol $o "$EVQ" 4:9 1:7 ) & local P1=$!; ( ev_p $pol $o "$EVQ" 8:0 8:3 8:5 8:7 8:9 4:5 8:1 8:4 1:8 4:1 ) & local P2=$!
    wait $P1 $P2
  else
    ( ev_t $pol $o "$EVQ" 0 1 2 3 4 5 6 7 8 9 ) & local P1=$!; ( ev_p $pol $o "$EVQ" 8:0 8:3 8:5 8:7 8:9 4:5 ) & local P2=$!; ( ev_p $pol $o "$EVQ" 4:9 1:7 8:1 8:4 1:8 4:1 ) & local P3=$!
    wait $P1 $P2 $P3
  fi; }
until grep -q "검출 미리 계산 끝" outputs/piper/objfeat_cache.log 2>/dev/null; do sleep 60; done
log "[1] bo1 학습 (ba1 에서, 절대 목표 + 물체 위치, 20000스텝)"
touch outputs/piper/TRAINING
$PY train_lora.py --policy outputs/piper_ba1_model/merged --data $DATA $TR --abs-pos --objfeat --dagger-frac 0.3 --steps 20000 --lr 5e-5 \
  --eval-every 2500 --save-every 4000 --out outputs/piper_bo1_model 2>&1 | grep --line-buffered -E '에피소드 학습|물체 위치|val_loss|모델 저장|Traceback|Error'
rm -f outputs/piper/TRAINING
B=outputs/piper_bo1_model/merged
[ -f $B/objfeat ] || { log "bo1 모델 없음 — 멈춤"; exit 1; }
log "[2] bo1 28항목"
all28 $B piper_bo1h
board piper_bo1h "블록 위치 입력 bo1 (새 배치 20장면, 지연 11)"
eval "$($PY piper_sim/plan_dagger.py --base piper_bo1h --start 3000)"
log "[3] bo1 교정 시범 — 단독 30:[${T30}] 20:[${T20}] 10:[${T10}] / 전환 30:[${P30}] 20:[${P20}] 10:[${P10}] / 재개 30:[${R30}] 20:[${R20}] 10:[${R10}]"
OUTD=data/piper_blk_dagger_bo1
D="--policy $B $EVQ --strategy keep --out $OUTD"
dag_t(){ for k in 30 20 10; do v=T$k; e=E$k; [ -n "${!v}" ] && $PY piper_sim/dagger_piper.py $D --tasks ${!v} --episodes ${!e} --seed $((71 + k)); done; }
dag_p(){ for k in 30 20 10; do v=P$k; e=E$k; [ -n "${!v}" ] && $PY piper_sim/dagger_piper.py $D --pairs ${!v} --episodes ${!e} --seed $((72 + k)); done; }
dag_r(){ for k in 30 20 10; do v=R$k; e=E$k; [ -n "${!v}" ] && $PY piper_sim/dagger_piper.py $D --pairs ${!v} --resume --episodes ${!e} --seed $((73 + k)); done; }
if [ "$(other)" -gt 0 ]; then
  ( dag_t > outputs/piper/dagger_bo1_t.log 2>&1; dag_r > outputs/piper/dagger_bo1_r.log 2>&1 ) & P1=$!
  ( dag_p > outputs/piper/dagger_bo1_p.log 2>&1 ) & P2=$!
  wait $P1 $P2
else
  ( dag_t > outputs/piper/dagger_bo1_t.log 2>&1 ) & P1=$!
  ( dag_p > outputs/piper/dagger_bo1_p.log 2>&1 ) & P2=$!
  ( dag_r > outputs/piper/dagger_bo1_r.log 2>&1 ) & P3=$!
  wait $P1 $P2 $P3
fi
log "교정 시범 끝: $(ls $OUTD/episodes 2>/dev/null | wc -l)"
$PY piper_sim/trim_stalls.py $OUTD ${OUTD}_t
for s in 0 1 2 3; do ( $PY objfeat_cache.py --data ${OUTD}_t --shard $s --nshard 4 > /dev/null 2>&1 ) & done; wait
log "[4] bo2 학습 (bo1 에서, + bo1 교정 시범, 16000스텝)"
touch outputs/piper/TRAINING
$PY train_lora.py --policy $B --data $DATA ${OUTD}_t $TR --abs-pos --objfeat --dagger-frac 0.35 --steps 16000 --lr 3e-5 \
  --eval-every 2000 --save-every 4000 --out outputs/piper_bo2_model 2>&1 | grep --line-buffered -E '에피소드 학습|물체 위치|val_loss|모델 저장|Traceback|Error'
rm -f outputs/piper/TRAINING
B2=outputs/piper_bo2_model/merged
[ -f $B2/objfeat ] || { log "bo2 모델 없음 — 멈춤"; exit 1; }
log "[5] bo2 28항목"
all28 $B2 piper_bo2h
board piper_bo2h "블록 위치 입력 bo2 (새 배치 20장면, 지연 11)"
log "끝"
