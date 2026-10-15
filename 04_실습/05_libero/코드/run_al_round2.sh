#!/usr/bin/env bash
# align 2라운드 (10/14): al1 28항목이 끝나면 → 95% 못 넘는 항목 교정 시범 (al1, align 으로 넘겨받기, 지연 9)
#   → 검출·사진 특징 계산 → al2 (al1 에서, align 시범 + 교정 시범 전부) → 28항목.  끝나면 같은 꼴로 al3 ...
set -u
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
ngpu(){ nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -vc rustdesk; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --log-every 500"
DATA="data/piper_al_normal_t data/piper_al_wide_normal_t data/piper_al_switch_t data/piper_al_recovery_t data/piper_al_dagger_bo1_t"
ev_t(){ local pol=$1 o=$2; shift 2; for t in "$@"; do $PY piper_sim/eval_piper.py --policy $pol --task-a $t --strategy none --episodes 20 --start-episode 1000 $EVQ --out outputs/${o}_forget 2>&1 | grep -E "$F"; done; }
ev_p(){ local pol=$1 o=$2; shift 2; for p in "$@"; do $PY piper_sim/eval_piper.py --policy $pol --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy keep --episodes 20 --start-episode 1000 $EVQ --out outputs/${o}_switch 2>&1 | grep -E "$F"; done; }
board(){ $PY scoreboard.py --blocks --forget outputs/$1_forget --switch outputs/$1_switch --name "$2" --png outputs/media/score_$1.png --md outputs/piper/score_$1.md | tail -1
         $PY scoreboard.py --blocks --loose --forget outputs/$1_forget --switch outputs/$1_switch --name "$2 판 안 기준" --png outputs/media/score_$1_loose.png --md outputs/piper/score_$1_loose.md | tail -1; }
all28(){ local pol=$1 o=$2
  until [ "$(ngpu)" -le 0 ]; do sleep 60; done
  ( ev_t $pol $o 0 1 2 3 4 ) & local P1=$!; ( ev_t $pol $o 5 6 7 8 9 ) & local P2=$!
  ( ev_p $pol $o 8:0 8:3 8:5 8:7 8:9 4:5 ) & local P3=$!; ( ev_p $pol $o 4:9 1:7 8:1 8:4 1:8 4:1 ) & local P4=$!
  wait $P1 $P2 $P3 $P4; }

round(){   # round 이전모델번호 새모델번호 교정시범장면시작
  local i=$1 j=$2 start=$3
  local B=outputs/piper_al${i}_model/merged OUTD=data/piper_al_dagger_al${i}
  eval "$($PY piper_sim/plan_dagger.py --base piper_al${i}h --start $start)"
  log "[al$i 교정 시범] 단독 30:[${T30}] 20:[${T20}] 10:[${T10}] / 전환 30:[${P30}] 20:[${P20}] 10:[${P10}] / 재개 30:[${R30}] 20:[${R20}] 10:[${R10}]"
  local D="--policy $B $EVQ --strategy keep --max-policy 350 --out $OUTD"
  dag_t(){ for k in 30 20 10; do v=T$k; e=E$k; [ -n "${!v}" ] && $PY piper_sim/dagger_piper.py $D --tasks ${!v} --episodes ${!e} --seed $((81 + k + 10 * i)); done; }
  dag_p(){ for k in 30 20 10; do v=P$k; e=E$k; [ -n "${!v}" ] && $PY piper_sim/dagger_piper.py $D --pairs ${!v} --episodes ${!e} --seed $((82 + k + 10 * i)); done; }
  dag_r(){ for k in 30 20 10; do v=R$k; e=E$k; [ -n "${!v}" ] && $PY piper_sim/dagger_piper.py $D --pairs ${!v} --resume --episodes ${!e} --seed $((83 + k + 10 * i)); done; }
  ( dag_t > outputs/piper/dagger_al${i}_t.log 2>&1 ) & local P1=$!
  ( dag_p > outputs/piper/dagger_al${i}_p.log 2>&1 ) & local P2=$!
  ( dag_r > outputs/piper/dagger_al${i}_r.log 2>&1 ) & local P3=$!
  wait $P1 $P2 $P3
  $PY piper_sim/trim_stalls.py $OUTD ${OUTD}_t
  for s in 0 1 2; do ( $PY objfeat_cache.py --data ${OUTD}_t --shard $s --nshard 3 > /dev/null 2>&1 ) & done; wait
  ( $PY vis_cache.py build --policy $B --data ${OUTD}_t --out cache/vis --batch 8 --stride 3 --shard 0 --nshard 2 2>&1 | grep -E "끝|Error|Traceback" ) & local Q1=$!
  ( $PY vis_cache.py build --policy $B --data ${OUTD}_t --out cache/vis --batch 8 --stride 3 --shard 1 --nshard 2 2>&1 | grep -E "끝|Error|Traceback" ) & local Q2=$!
  wait $Q1 $Q2
  DATA="$DATA ${OUTD}_t"
  log "[al$j 학습] al$i 에서, 교정 시범 $(ls $OUTD/episodes | wc -l)개 더함"
  $PY train_lora.py --policy $B --data $DATA $TR --abs-pos --objfeat --vis-cache cache/vis --frame-stride 3 \
    --dagger-frac 0.35 --batch-size 8 --grad-accum 2 --steps 10000 --lr 3e-5 --eval-every 2500 --save-every 5000 \
    --out outputs/piper_al${j}_model 2>&1 | grep --line-buffered -E '에피소드 학습|val_loss|모델 저장|Traceback|Error'
  [ -f outputs/piper_al${j}_model/merged/objfeat ] || { log "al$j 모델 없음 — 멈춤"; exit 1; }
  log "[al$j 28항목]"
  all28 outputs/piper_al${j}_model/merged piper_al${j}h
  board piper_al${j}h "align al$j (새 배치 20장면, 지연 9)"
}

until grep -q "^=== .* 끝$" outputs/piper/al_round.log 2>/dev/null; do sleep 300; done
round 1 2 3500
round 2 3 3600
round 3 4 3700
log "끝"
