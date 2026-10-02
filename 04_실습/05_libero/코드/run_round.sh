#!/usr/bin/env bash
# 한 라운드: 95% 미만 항목 DAgger(실제 배치 설정 그대로) → 다음 모델 학습 → A2C2 → 28개 항목 평가 (2026-10-02)
#   ./run_round.sh PREV NEXT EVALPREFIX_BASE EVALPREFIX_A2C2 A2C2DIR_PREV
#   예: ./run_round.sh v6c v6d v6ch v6cah outputs/a2c2_v6c   (평가는 학습에 안 쓴 새 배치, 장면 1000번대)
# GPU 프로세스 4개, 메모리: 학습은 혼자.
set -u
cd "$(dirname "$0")"
PREV=$1; NEXT=$2; EB=$3; EA=$4; HEAD=$5
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
gpu_n(){ ps -eo comm,args | awk '$1=="python" && /switch_experiment\.py|dagger_v6\.py|a2c2\.py|train_lora\.py|teacher_audit\.py/' | wc -l; }

eval "$($PY plan_round.py --base $EB --a2c2 $EA --log outputs/v6/run_post_${PREV}.log)"
log "설정: n_act=$NA, A2C2 사용=$USE_A2C2 (평균 $MEAN_BASE → $MEAN_A2C2), 보정 차원=$DIMS"
log "95% 미만: 단독 [$TASKS] 전환 [$PAIRS] 재개 [$RESUME]"
POL=outputs/${PREV}_model/merged
AOPT=""; [ "$USE_A2C2" = "1" ] && AOPT="--a2c2 $HEAD --a2c2-dims $DIMS"
D="--policy $POL --latency-steps 11 --ttrtc --strategy keep --n-action-steps $NA $AOPT --episodes 0-19,50-69 --out data/dagger_$NEXT"

log "[1] DAgger (실제 배치 설정: 지연 11, 지연 흉내내기, 멈춤 없는 전환, 보정)"
JOBS=()
[ -n "$TASKS" ] && JOBS+=("--tasks $TASKS")
if [ -n "$PAIRS" ]; then
  IFS=',' read -ra PP <<< "$PAIRS"; h=$(( (${#PP[@]} + 1) / 2 ))
  JOBS+=("--pairs $(IFS=,; echo "${PP[*]:0:$h}")")
  [ ${#PP[@]} -gt $h ] && JOBS+=("--pairs $(IFS=,; echo "${PP[*]:$h}")")
fi
[ -n "$RESUME" ] && JOBS+=("--pairs $RESUME --resume")
k=0
for J in "${JOBS[@]}"; do
  while [ "$(gpu_n)" -ge 4 ]; do sleep 30; done
  k=$((k+1))
  setsid $PY dagger_v6.py $D $J --seed $((100 + k)) > outputs/v6/dagger_${NEXT}_$k.log 2>&1 &
  sleep 20
done
wait
log "DAgger 끝: $(ls data/dagger_$NEXT/episodes 2>/dev/null | wc -l) 시범"

while [ "$(gpu_n)" -gt 0 ]; do sleep 60; done        # 학습은 혼자 (메모리 23GB)
log "[2] $NEXT 학습 ($PREV 에서 이어서, 동작 통계는 $PREV 것 그대로)"
DATA="data/v6_normal data/v6_switch data/v6_dagger_lat $(ls -d data/dagger_v6* data/v7_* 2>/dev/null | tr '\n' ' ')"   # v7_*: 고친 시범 프로그램, 새 무작위 배치(장면 2000번대)
$PY train_lora.py --policy $POL --data $DATA --full-expert --aug --balance --dagger-frac 0.3 --workers 0 --rtc-max-delay 14 --ema 0.999 \
  --steps 24000 --batch-size 4 --grad-accum 2 --lr 3e-5 --eval-every 3000 --save-every 6000 --log-every 500 \
  --out outputs/${NEXT}_model 2>&1 | grep --line-buffered -E '지연 흉내|교정 시범 비율|에피소드 학습|val_loss|모델 저장|Traceback|Error'
NPOL=outputs/${NEXT}_model/merged
[ -f $NPOL/model.safetensors ] || { log "$NEXT 모델이 없다 — 멈춤"; exit 1; }
SAN=$($PY switch_experiment.py --policy $NPOL --task-a 8 --strategy none --episodes 5 --start-episode 0 --out outputs/${NEXT}_sanity 2>&1 \
  | grep SUMMARY | sed -E 's/.*"a_success_rate": ([0-9.]+).*/\1/')
log "쉬운 점검 T8 (학습 배치, 지연 없음, 5장면): $SAN"
python3 -c "import sys; sys.exit(0 if float('${SAN:-0}') >= 0.4 else 1)" || { log "점검 실패 — 멈춤"; exit 1; }

log "[3] A2C2 데이터 ($NEXT) + $NEXT 혼자 평가"
EV="--episodes 10 --start-episode 1000 --latency-steps 11 --ttrtc --n-action-steps $NA"   # 학습에 안 쓴 새 배치
( $PY a2c2.py gen --policy $NPOL --ttrtc --data $DATA --frac 0.5 --stride 16 --batch 8 --out data/a2c2_$NEXT 2>&1 | grep -E "끝|Traceback|Error" ) &
( for t in 0 1 2 3 4 5 6 7 8 9; do $PY switch_experiment.py --policy $NPOL --task-a $t --strategy none $EV --out outputs/${NEXT}h_forget 2>&1 | grep -E "$F"; done ) &
( for p in 8:0 8:3 8:5 8:7 8:9 4:5 4:9 1:7 8:1 8:4 1:8 4:1; do $PY switch_experiment.py --policy $NPOL --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy keep $EV --out outputs/${NEXT}h_switch 2>&1 | grep -E "$F"; done ) &
wait
log "[4] A2C2 학습"
$PY a2c2.py train --gen data/a2c2_$NEXT --out outputs/a2c2_$NEXT --steps 30000 --batch 64 --workers 4 2>&1 \
  | grep --line-buffered -E '보정 네트워크|"step": [0-9]*0000,|저장|Traceback|Error'
log "[5] $NEXT + A2C2 평가"
EVA="$EV --a2c2 outputs/a2c2_$NEXT --a2c2-dims $DIMS"
( for t in 0 1 2 3 4 5 6 7 8 9; do $PY switch_experiment.py --policy $NPOL --task-a $t --strategy none $EVA --out outputs/${NEXT}ah_forget 2>&1 | grep -E "$F"; done ) &
( for p in 8:0 8:3 8:5 8:7 8:9 4:5; do $PY switch_experiment.py --policy $NPOL --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy keep $EVA --out outputs/${NEXT}ah_switch 2>&1 | grep -E "$F"; done ) &
( for p in 4:9 1:7 8:1 8:4 1:8 4:1; do $PY switch_experiment.py --policy $NPOL --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy keep $EVA --out outputs/${NEXT}ah_switch 2>&1 | grep -E "$F"; done ) &
wait
for X in ${NEXT}h ${NEXT}ah; do
  $PY scoreboard.py --forget outputs/${X}_forget --switch outputs/${X}_switch --name "$X 지연11" --png outputs/media/score_$X.png \
    --md outputs/v6/score_$X.md | tail -1
done
log "라운드 끝 ($NEXT)"
