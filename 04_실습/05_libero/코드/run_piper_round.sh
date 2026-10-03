#!/usr/bin/env bash
# PiPER 한 라운드 (10/3): PREV 모델로 95% 미만 항목 DAgger(실제 배치 조건) → NEXT 학습(PREV 에서) → 새 배치 평가
#   → NEXT 의 A2C2(매 스텝 보정) 학습 → NEXT+A2C2 평가
#   ./run_piper_round.sh piper_s1 piper_s2 2100-2129
set -u
cd "$(dirname "$0")"
PREV=$1; NEXT=$2; EPS=${3:-2100-2129}
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
POL=outputs/${PREV}_model/merged
EVQ="--latency-steps 11 --ttrtc --n-action-steps 1"
# 1) 95% 미만 항목 (PREV 새 배치 평가 결과) — 성적이 낮을수록 교정 시범 장면을 많이 (30/20/10)
eval "$($PY piper_sim/plan_dagger.py --base ${PREV}h --start ${EPS%-*})"
log "[1] $PREV 95% 미만 — 단독 30:[${T30}] 20:[${T20}] 10:[${T10}] / 전환 30:[${P30}] 20:[${P20}] 10:[${P10}] / 재개 30:[${R30}] 20:[${R20}] 10:[${R10}]"
OUTD=data/piper_dagger_${NEXT}
D="--policy $POL $EVQ --strategy keep --out $OUTD"
dag_t(){ for k in 30 20 10; do v=T$k; e=E$k; [ -n "${!v}" ] && $PY piper_sim/dagger_piper.py $D --tasks ${!v} --episodes ${!e} --seed $((11 + k)); done; }
dag_p(){ for k in 30 20 10; do v=P$k; e=E$k; [ -n "${!v}" ] && $PY piper_sim/dagger_piper.py $D --pairs ${!v} --episodes ${!e} --seed $((12 + k)); done; }
dag_r(){ for k in 30 20 10; do v=R$k; e=E$k; [ -n "${!v}" ] && $PY piper_sim/dagger_piper.py $D --pairs ${!v} --resume --episodes ${!e} --seed $((13 + k)); done; }
( dag_t > outputs/piper/dagger_${NEXT}_t.log 2>&1 ) &
( dag_p > outputs/piper/dagger_${NEXT}_p.log 2>&1 ) &
( dag_r > outputs/piper/dagger_${NEXT}_r.log 2>&1 ) &
wait
log "DAgger 끝: $(ls $OUTD/episodes 2>/dev/null | wc -l) 시범"
[ -d $OUTD/episodes ] && $PY piper_sim/trim_stalls.py $OUTD ${OUTD}_t
# 2) NEXT 학습 (PREV 에서, 시범 전부 + 교정 시범)
DATA="data/piper_normal_t data/piper_switch_t $(ls -d data/piper_normal2_t data/piper_switch2_t data/piper_dagger_*_t 2>/dev/null | tr '\n' ' ')"
log "[2] $NEXT 학습: $DATA"
touch outputs/piper/TRAINING
$PY train_lora.py --policy $POL --data $DATA --full-expert --aug --balance --dagger-frac 0.3 --workers 0 --rtc-max-delay 14 \
  --ema 0.999 --frame-stride 2 --steps 16000 --batch-size 4 --grad-accum 2 --lr 3e-5 --eval-every 2000 --save-every 4000 \
  --log-every 500 --out outputs/${NEXT}_model 2>&1 | grep --line-buffered -E '에피소드 학습|val_loss|모델 저장|Traceback|Error'
rm -f outputs/piper/TRAINING
NPOL=outputs/${NEXT}_model/merged
[ -f $NPOL/model.safetensors ] || { log "$NEXT 모델 없음 — 멈춤"; exit 1; }
# 3) 새 배치 평가 ‖ A2C2 데이터
EV="--episodes 20 --start-episode 1000 $EVQ"
run_t(){ local o=$1; shift; for t in "$@"; do $PY piper_sim/eval_piper.py --policy $NPOL --task-a $t --strategy none $EV $AOPT --out outputs/${o}_forget 2>&1 | grep -E "$F"; done; }
run_p(){ local o=$1; shift; for p in "$@"; do $PY piper_sim/eval_piper.py --policy $NPOL --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy keep $EV $AOPT --out outputs/${o}_switch 2>&1 | grep -E "$F"; done; }
log "[3] $NEXT 새 배치 평가 + A2C2 데이터"
AOPT=""
( run_t ${NEXT}h 0 1 2 3 4 5 6 7 8 9 ) & ( run_p ${NEXT}h 8:0 8:3 8:5 8:7 8:9 4:5 ) & ( run_p ${NEXT}h 4:9 1:7 8:1 8:4 1:8 4:1 ) &
( $PY a2c2.py gen --policy $NPOL --ttrtc --data $DATA --frac 0.5 --stride 16 --batch 8 --out data/a2c2_${NEXT} 2>&1 | grep -E "끝|Traceback|Error" ) &
wait
$PY scoreboard.py --forget outputs/${NEXT}h_forget --switch outputs/${NEXT}h_switch --name "$NEXT (새 배치 20장면, 지연 11)" \
  --png outputs/media/score_${NEXT}.png --md outputs/piper/score_${NEXT}.md | tail -1
log "[4] A2C2 학습"
$PY a2c2.py train --gen data/a2c2_${NEXT} --out outputs/a2c2_${NEXT} --steps 30000 --batch 64 --workers 4 2>&1 | grep -E "끝|val|Traceback|Error" | tail -3
log "[5] $NEXT + A2C2 새 배치 평가"
AOPT="--a2c2 outputs/a2c2_${NEXT}"
( run_t ${NEXT}ah 0 1 2 3 4 ) & ( run_t ${NEXT}ah 5 6 7 8 9 ) & ( run_p ${NEXT}ah 8:0 8:3 8:5 8:7 8:9 4:5 ) & ( run_p ${NEXT}ah 4:9 1:7 8:1 8:4 1:8 4:1 ) &
wait
$PY scoreboard.py --forget outputs/${NEXT}ah_forget --switch outputs/${NEXT}ah_switch --name "$NEXT + A2C2 (새 배치 20장면, 지연 11)" \
  --png outputs/media/score_${NEXT}a.png --md outputs/piper/score_${NEXT}a.md | tail -1
log "라운드 끝"
