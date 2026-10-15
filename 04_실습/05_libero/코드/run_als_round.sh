#!/usr/bin/env bash
# 상태를 동작 전문가 쪽에도 (10/15 밤, ttrtc.enable_sufstate): alr1 은 손 기준 물체 위치를 2cm 바꿔도 동작이 0.1cm 도 안 바뀌었다.
#   ① alr1 28항목(이미 돌고 있는 평가 서브셸 4개)이 끝나면 점수표 → al1 교정 시범 135개 정리
#   ② als1 학습 (alr1 에서, --sufstate, GPU 혼자) → 민감도 확인 → 28항목
#   ③ als1 교정 시범 (95% 못 넘는 항목, 절반 규모) → als2 → 28항목 → als3 ...
set -u
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --log-every 500 --abs-pos --objfeat --feat relfeat --vis-cache cache/vis --frame-stride 3 --sufstate"
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
train(){ local init=$1 out=$2 steps=$3 lr=$4 frac=$5; shift 5
  $PY train_lora.py --policy $init --data $DATA "$@" $TR --dagger-frac $frac --batch-size 8 --grad-accum 2 --steps $steps --lr $lr \
    --eval-every 2500 --save-every 5000 --out $out 2>&1 | grep --line-buffered -E '에피소드 학습|상태를 동작|val_loss|모델 저장|Traceback|Error'; }
sens(){ CUDA_VISIBLE_DEVICES= timeout 3600 $PY sens_check.py $1 2>&1 | grep -E "쥐기"; }

for p in 301727 301728 301729 301731; do while kill -0 $p 2>/dev/null; do sleep 60; done; done
log "alr1 28항목 끝"
board piper_alr1h "손 기준 물체 위치 alr1 (새 배치 20장면, 지연 9)"
prep_data data/piper_al_dagger_al1 outputs/piper_alr1_model/merged
DATA="$DATA data/piper_al_dagger_al1_t"
log "[1] als1 학습 (alr1 에서, 상태를 동작 쪽에도, 15000스텝)"
train outputs/piper_alr1_model/merged outputs/piper_als1_model 15000 5e-5 0.25
B=outputs/piper_als1_model/merged
[ -f $B/sufstate ] || { log "als1 모델 없음 — 멈춤"; exit 1; }
log "[2] als1 민감도 (빨강 블록 위치 +2cm → 목표 위치가 따라가나)"; sens $B
log "[3] als1 28항목"
all28 $B piper_als1h
board piper_als1h "상태를 동작 쪽에도 als1 (새 배치 20장면, 지연 9)"
i=1
while [ $i -le 3 ]; do
  j=$((i + 1)); B=outputs/piper_als${i}_model/merged; OUTD=data/piper_al_dagger_als${i}
  eval "$($PY piper_sim/plan_dagger.py --base piper_als${i}h --start $((3700 + 100 * i)))"
  log "[als$i 교정 시범] 단독 30:[${T30}] 20:[${T20}] 10:[${T10}] / 전환 30:[${P30}] 20:[${P20}] / 재개 30:[${R30}] 20:[${R20}]"
  DG="--policy $B $EVQ --strategy keep --max-policy 350 --out $OUTD"
  dag_t(){ for k in 30 20 10; do v=T$k; e=E$k; [ -n "${!v}" ] && $PY piper_sim/dagger_piper.py $DG --tasks ${!v} --episodes ${!e} --seed $((151 + k + 10 * i)); done; }
  dag_p(){ for k in 30 20 10; do v=P$k; e=E$k; [ -n "${!v}" ] && $PY piper_sim/dagger_piper.py $DG --pairs ${!v} --episodes ${!e} --seed $((152 + k + 10 * i)); done; }
  dag_r(){ for k in 30 20 10; do v=R$k; e=E$k; [ -n "${!v}" ] && $PY piper_sim/dagger_piper.py $DG --pairs ${!v} --resume --episodes ${!e} --seed $((153 + k + 10 * i)); done; }
  ( dag_t > outputs/piper/dagger_als${i}_t.log 2>&1 ) & D1=$!
  ( dag_p > outputs/piper/dagger_als${i}_p.log 2>&1 ) & D2=$!
  ( dag_r > outputs/piper/dagger_als${i}_r.log 2>&1 ) & D3=$!
  wait $D1 $D2 $D3
  log "als$i 교정 시범 끝: $(ls $OUTD/episodes | wc -l)"
  prep_data $OUTD $B
  DATA="$DATA ${OUTD}_t"
  log "[als$j 학습] als$i 에서"
  train $B outputs/piper_als${j}_model 10000 3e-5 0.35
  [ -f outputs/piper_als${j}_model/merged/sufstate ] || { log "als$j 모델 없음 — 멈춤"; exit 1; }
  log "[als$j 28항목]"
  all28 outputs/piper_als${j}_model/merged piper_als${j}h
  board piper_als${j}h "상태를 동작 쪽에도 als$j (새 배치 20장면, 지연 9)"
  i=$j
done
log "끝"
