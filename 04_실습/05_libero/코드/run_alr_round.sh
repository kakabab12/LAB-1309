#!/usr/bin/env bash
# 손 기준 물체 위치 입력 라운드 (10/15, relfeat.py)
#   ① 정렬 보조 시험(run_assist_now.sh)이 끝나면 alr1 학습 (al1 에서, 정면 16칸 = 손 기준 m, 통계·열 다시)
#      그동안 al1 교정 시범 2개 (CUDA 그래프 없이, 학습과 GPU 를 나눠 씀)
#   ② alr1 28항목 → ③ 교정 시범 더해 alr2 (alr1 에서) → 28항목
set -u
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
ngpu(){ nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -vc rustdesk; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --log-every 500 --abs-pos --objfeat --feat relfeat --vis-cache cache/vis --frame-stride 3"
DATA="data/piper_al_normal_t data/piper_al_wide_normal_t data/piper_al_switch_t data/piper_al_recovery_t data/piper_al_dagger_bo1_t"
ev_t(){ local pol=$1 o=$2; shift 2; for t in "$@"; do $PY piper_sim/eval_piper.py --policy $pol --task-a $t --strategy none --episodes 20 --start-episode 1000 $EVQ --out outputs/${o}_forget 2>&1 | grep -E "$F"; done; }
ev_p(){ local pol=$1 o=$2; shift 2; for p in "$@"; do $PY piper_sim/eval_piper.py --policy $pol --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy keep --episodes 20 --start-episode 1000 $EVQ --out outputs/${o}_switch 2>&1 | grep -E "$F"; done; }
board(){ $PY scoreboard.py --blocks --forget outputs/$1_forget --switch outputs/$1_switch --name "$2" --png outputs/media/score_$1.png --md outputs/piper/score_$1.md | tail -1
         $PY scoreboard.py --blocks --loose --forget outputs/$1_forget --switch outputs/$1_switch --name "$2 판 안 기준" --png outputs/media/score_$1_loose.png --md outputs/piper/score_$1_loose.md | tail -1; }
dag_running(){ ps -eo args | grep -cE "^\.venv/bin/python piper_sim/dagger_piper\.py"; }
all28(){ local pol=$1 o=$2
  if [ "$(dag_running)" -gt 0 ]; then
    log "  교정 시범이 돌아 평가 2개"
    ( ev_t $pol $o 0 1 2 3 4 5 6 7 8 9 ) & local P1=$!; ( ev_p $pol $o 8:0 8:3 8:5 8:7 8:9 4:5 4:9 1:7 8:1 8:4 1:8 4:1 ) & local P2=$!
    wait $P1 $P2
  else
    ( ev_t $pol $o 0 1 2 3 4 ) & local P1=$!; ( ev_t $pol $o 5 6 7 8 9 ) & local P2=$!
    ( ev_p $pol $o 8:0 8:3 8:5 8:7 8:9 4:5 ) & local P3=$!; ( ev_p $pol $o 4:9 1:7 8:1 8:4 1:8 4:1 ) & local P4=$!
    wait $P1 $P2 $P3 $P4
  fi; }

until grep -q "^=== .* 끝$" outputs/piper/assist_now.log 2>/dev/null; do sleep 60; done
OUTD=data/piper_al_dagger_al1
DG="--policy outputs/piper_al1_model/merged --latency-steps 9 --ttrtc --n-action-steps 1 --strategy keep --max-policy 350 --out $OUTD"
( $PY piper_sim/dagger_piper.py $DG --tasks 0 1 2 3 4 5 6 7 8 9 --episodes 3500-3519 --seed 131 > outputs/piper/dagger_al1_t.log 2>&1
  $PY piper_sim/dagger_piper.py $DG --pairs 8:5,8:7,8:9,4:5,4:9,1:7 --resume --episodes 3500-3509 --seed 133 > outputs/piper/dagger_al1_r.log 2>&1 ) & D1=$!
( $PY piper_sim/dagger_piper.py $DG --pairs 8:0,8:3,8:5,8:7,8:9,4:5,4:9,1:7,8:1,8:4,1:8,4:1 --episodes 3500-3509 --seed 132 > outputs/piper/dagger_al1_p.log 2>&1 ) & D2=$!
log "[1] alr1 학습 (al1 에서, 손 기준 물체 위치) ‖ al1 교정 시범 2개"
$PY train_lora.py --policy outputs/piper_al1_model/merged --data $DATA $TR --reset-feat \
  --dagger-frac 0.25 --batch-size 8 --grad-accum 2 --steps 15000 --lr 5e-5 --eval-every 2500 --save-every 5000 \
  --out outputs/piper_alr1_model 2>&1 | grep --line-buffered -E '에피소드 학습|물체 위치|val_loss|모델 저장|Traceback|Error'
B=outputs/piper_alr1_model/merged
[ -f $B/relfeat ] || { log "alr1 모델 없음 — 멈춤"; exit 1; }
log "[2] alr1 28항목 (지연 9)"
all28 $B piper_alr1h
board piper_alr1h "손 기준 물체 위치 alr1 (새 배치 20장면, 지연 9)"
wait $D1 $D2
log "al1 교정 시범 끝: $(ls $OUTD/episodes 2>/dev/null | wc -l)"
$PY piper_sim/trim_stalls.py $OUTD ${OUTD}_t
for s in 0 1 2; do ( $PY objfeat_cache.py --data ${OUTD}_t --shard $s --nshard 3 > /dev/null 2>&1 ) & done; wait
$PY relfeat.py --data ${OUTD}_t
( $PY vis_cache.py build --policy $B --data ${OUTD}_t --out cache/vis --batch 8 --stride 3 --shard 0 --nshard 2 2>&1 | grep -E "끝|Error|Traceback" ) & Q1=$!
( $PY vis_cache.py build --policy $B --data ${OUTD}_t --out cache/vis --batch 8 --stride 3 --shard 1 --nshard 2 2>&1 | grep -E "끝|Error|Traceback" ) & Q2=$!
wait $Q1 $Q2
log "[3] alr2 학습 (alr1 에서, + al1 교정 시범)"
$PY train_lora.py --policy $B --data $DATA ${OUTD}_t $TR \
  --dagger-frac 0.35 --batch-size 8 --grad-accum 2 --steps 10000 --lr 3e-5 --eval-every 2500 --save-every 5000 \
  --out outputs/piper_alr2_model 2>&1 | grep --line-buffered -E '에피소드 학습|val_loss|모델 저장|Traceback|Error'
B2=outputs/piper_alr2_model/merged
[ -f $B2/relfeat ] || { log "alr2 모델 없음 — 멈춤"; exit 1; }
log "[4] alr2 28항목"
all28 $B2 piper_alr2h
board piper_alr2h "손 기준 물체 위치 alr2 (새 배치 20장면, 지연 9)"
log "끝"
