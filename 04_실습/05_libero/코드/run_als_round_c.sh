#!/usr/bin/env bash
# run_als_round.sh 이어 하기 (10/16 10:40): als1 교정 시범 3개 중 단독(T4~T9)·일 바꾸기가 GPU 메모리 부족으로 죽었다
#   (CUDA 그래프를 장면마다 새로 녹화해 한 프로세스가 3.9GB 까지 커짐 + 놓기 보조 시험). fastinfer 는 그래프 3개까지로 고침.
#   ① 빠진 교정 시범을 다시 (CUDA 그래프 없이) ‖ 돌아오기 교정 시범(서브셸 919731)은 그대로 → ② 정리 → als2 → 28항목 → als3 → als4
set -u
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
DGQ="--latency-steps 9 --ttrtc --n-action-steps 1"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --log-every 500 --abs-pos --objfeat --feat relfeat --vis-cache cache/vis --frame-stride 3 --sufstate"
DATA="data/piper_al_normal_t data/piper_al_wide_normal_t data/piper_al_switch_t data/piper_al_recovery_t data/piper_al_dagger_bo1_t data/piper_al_dagger_al1_t"
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

B=outputs/piper_als1_model/merged; OUTD=data/piper_al_dagger_als1
DG="--policy $B $DGQ --strategy keep --max-policy 350 --out $OUTD"
log "[als1 교정 시범 다시] 단독 T4~T9 ‖ 일 바꾸기 12쌍 (15장면씩, CUDA 그래프 없이)"
( $PY piper_sim/dagger_piper.py $DG --tasks 4 5 6 7 8 9 --episodes 3800-3814 --seed 191 > outputs/piper/dagger_als1_t2.log 2>&1 ) & A=$!
( $PY piper_sim/dagger_piper.py $DG --pairs 8:0,8:3,8:5,8:7,8:9,4:5,4:9,1:7,8:1,8:4,1:8,4:1 --episodes 3800-3814 --seed 192 > outputs/piper/dagger_als1_p2.log 2>&1 ) & Bp=$!
wait $A $Bp
while kill -0 919731 2>/dev/null; do sleep 60; done
log "als1 교정 시범 끝: $(ls $OUTD/episodes | wc -l)"
prep_data $OUTD $B
DATA="$DATA ${OUTD}_t"
i=1
while [ $i -le 3 ]; do
  j=$((i + 1)); B=outputs/piper_als${i}_model/merged
  log "[als$j 학습] als$i 에서"
  train $B outputs/piper_als${j}_model 10000 3e-5 0.35
  [ -f outputs/piper_als${j}_model/merged/sufstate ] || { log "als$j 모델 없음 — 멈춤"; exit 1; }
  log "[als$j 28항목]"
  all28 outputs/piper_als${j}_model/merged piper_als${j}h
  board piper_als${j}h "상태를 동작 쪽에도 als$j (새 배치 20장면, 지연 9)"
  [ $j -ge 4 ] && break
  B=outputs/piper_als${j}_model/merged; OUTD=data/piper_al_dagger_als${j}
  eval "$($PY piper_sim/plan_dagger.py --base piper_als${j}h --start $((3800 + 100 * j)))"
  log "[als$j 교정 시범] 단독 30:[${T30}] 20:[${T20}] 10:[${T10}] / 전환 30:[${P30}] 20:[${P20}] / 재개 30:[${R30}] 20:[${R20}]"
  DG="--policy $B $DGQ --strategy keep --max-policy 350 --out $OUTD"
  dag_t(){ for k in 30 20 10; do v=T$k; e=E$k; [ -n "${!v}" ] && $PY piper_sim/dagger_piper.py $DG --tasks ${!v} --episodes ${!e} --seed $((201 + k + 10 * j)); done; }
  dag_p(){ for k in 30 20 10; do v=P$k; e=E$k; [ -n "${!v}" ] && $PY piper_sim/dagger_piper.py $DG --pairs ${!v} --episodes ${!e} --seed $((202 + k + 10 * j)); done; }
  dag_r(){ for k in 30 20 10; do v=R$k; e=E$k; [ -n "${!v}" ] && $PY piper_sim/dagger_piper.py $DG --pairs ${!v} --resume --episodes ${!e} --seed $((203 + k + 10 * j)); done; }
  ( dag_t > outputs/piper/dagger_als${j}_t.log 2>&1 ) & D1=$!
  ( dag_p > outputs/piper/dagger_als${j}_p.log 2>&1 ) & D2=$!
  ( dag_r > outputs/piper/dagger_als${j}_r.log 2>&1 ) & D3=$!
  wait $D1 $D2 $D3
  log "als$j 교정 시범 끝: $(ls $OUTD/episodes | wc -l)"
  prep_data $OUTD $B
  DATA="$DATA ${OUTD}_t"
  i=$j
done
log "끝"
