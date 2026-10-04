#!/usr/bin/env bash
# 블록 2차 piper_b2 (10/4): b1 의 95% 미만 항목 교정 시범(DAgger, 실제 배치 조건, 장면 2200~) + 2차 수집 시범
#   → b1 에서 이어 학습 → 새 배치 평가 → A2C2 → 평가.
#   b2 학습 동안 GPU 남는 자리에서: 비교용 b0n 평가(2), 반복 5단계 시험(1).
set -u
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 MUJOCO_GL=egl
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
until grep -q "=== .* 끝 (비교용" outputs/piper/blk_round1.log 2>/dev/null; do sleep 120; done
B1=outputs/piper_b1_model/merged
EVQ="--latency-steps 11 --ttrtc --n-action-steps 1"
# 교정 시범은 실제로 쓸 조합으로 모은다: A2C2 를 붙인 쪽이 평균이 높으면 붙여서
read -r _ M0 <<< "$($PY piper_sim/mean_score.py piper_b1h)"; read -r _ M1 <<< "$($PY piper_sim/mean_score.py piper_b1ah)"
if python3 -c "import sys; sys.exit(0 if $M1 > $M0 else 1)"; then BASE=piper_b1ah; DA="--a2c2 outputs/a2c2_piper_b1"; else BASE=piper_b1h; DA=""; fi
log "[1] 교정 시범: b1 평균 $M0, b1+A2C2 평균 $M1 → $BASE 의 95% 미만 항목"
eval "$($PY piper_sim/plan_dagger.py --base $BASE --start 2200)"
log "  단독 30:[${T30}] 20:[${T20}] 10:[${T10}] / 전환 30:[${P30}] 20:[${P20}] 10:[${P10}] / 재개 30:[${R30}] 20:[${R20}] 10:[${R10}]"
OUTD=data/piper_blk_dagger_b2
D="--policy $B1 $EVQ $DA --strategy keep --out $OUTD"
dag_t(){ for k in 30 20 10; do v=T$k; e=E$k; [ -n "${!v}" ] && $PY piper_sim/dagger_piper.py $D --tasks ${!v} --episodes ${!e} --seed $((31 + k)); done; }
dag_p(){ for k in 30 20 10; do v=P$k; e=E$k; [ -n "${!v}" ] && $PY piper_sim/dagger_piper.py $D --pairs ${!v} --episodes ${!e} --seed $((32 + k)); done; }
dag_r(){ for k in 30 20 10; do v=R$k; e=E$k; [ -n "${!v}" ] && $PY piper_sim/dagger_piper.py $D --pairs ${!v} --resume --episodes ${!e} --seed $((33 + k)); done; }
( dag_t > outputs/piper/dagger_b2_t.log 2>&1 ) & P1=$!
( dag_p > outputs/piper/dagger_b2_p.log 2>&1 ) & P2=$!
( dag_r > outputs/piper/dagger_b2_r.log 2>&1 ) & P3=$!
wait $P1 $P2 $P3
log "교정 시범 끝: $(ls $OUTD/episodes 2>/dev/null | wc -l)"
[ -d $OUTD/episodes ] && $PY piper_sim/trim_stalls.py $OUTD ${OUTD}_t
until grep -q "블록 2차 수집 끝" outputs/piper/blk_collect2.log 2>/dev/null; do sleep 120; done
while pgrep -f "train_lora[.]py .*piper_b0n_model" >/dev/null; do sleep 60; done    # 학습 2개를 겹치지 않는다 (메모리)
DATA="data/piper_blk_normal_t data/piper_blk_switch_t data/piper_blk_normal2_t data/piper_blk_switch2_t ${OUTD}_t"
log "[2] piper_b2 학습 (b1 에서): $DATA"
touch outputs/piper/TRAINING
( $PY train_lora.py --policy $B1 --data $DATA --full-expert --aug --balance --dagger-frac 0.3 --workers 0 --rtc-max-delay 14 \
  --ema 0.999 --frame-stride 2 --steps 16000 --batch-size 4 --grad-accum 2 --lr 3e-5 --eval-every 2000 --save-every 4000 \
  --log-every 500 --out outputs/piper_b2_model 2>&1 | grep --line-buffered -E '에피소드 학습|val_loss|모델 저장|Traceback|Error' ) & PT=$!
# 학습과 나란히: 비교용 b0n 평가 (2), 반복 5단계 시험 (1)
EV="--episodes 20 --start-episode 1000"
ev_t(){ local pol=$1 o=$2 opt=$3; shift 3; for t in "$@"; do $PY piper_sim/eval_piper.py --policy $pol --task-a $t --strategy none $EV $opt --out outputs/${o}_forget 2>&1 | grep -E "$F"; done; }
ev_p(){ local pol=$1 o=$2 opt=$3; shift 3; for p in "$@"; do $PY piper_sim/eval_piper.py --policy $pol --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy keep $EV $opt --out outputs/${o}_switch 2>&1 | grep -E "$F"; done; }
B0=outputs/piper_b0n_model/merged
if [ -f $B0/model.safetensors ]; then
  ( ev_t $B0 piper_b0nh "$EVQ" 0 1 2 3 4 5 6 7 8 9; ev_p $B0 piper_b0nh "$EVQ" 8:5 8:7 8:9 ) &
  ( ev_p $B0 piper_b0nh "$EVQ" 8:0 8:3 4:5 4:9 1:7 8:1 8:4 1:8 4:1 ) &
fi
# 반복 5단계: GPU 혼자 잰 생각 시간으로 지연을 정한다 (bench_split_alone.json)
L5=$(python3 -c "import json,math; d=json.load(open('outputs/piper/bench_split_alone.json')); print(math.ceil(d['total_5steps_ms']/50))" 2>/dev/null || echo 8)
log "  반복 5단계 시험: 지연 $L5 스텝 (b1, 같은 장면 1000~1019)"
NS="--latency-steps $L5 --ttrtc --n-action-steps 1 --num-steps 5"
( ev_t $B1 piper_b1ns5h "$NS" 1 4 6 8 9; ev_p $B1 piper_b1ns5h "$NS" 8:1 8:5 4:9 ) &
wait $PT
rm -f outputs/piper/TRAINING
POL=outputs/piper_b2_model/merged
[ -f $POL/model.safetensors ] || { log "b2 모델 없음 — 멈춤"; exit 1; }
while pgrep -f "eval_piper[.]py" >/dev/null; do sleep 60; done
[ -d outputs/piper_b0nh_forget ] && $PY scoreboard.py --forget outputs/piper_b0nh_forget --switch outputs/piper_b0nh_switch \
  --name "비교: 정상 시범만 b0n (새 배치 20장면, 지연 11)" --png outputs/media/score_piper_b0n.png --md outputs/piper/score_piper_b0n.md | tail -1
$PY scoreboard.py --forget outputs/piper_b1ns5h_forget --switch outputs/piper_b1ns5h_switch --name "b1 반복 5단계 (지연 $L5)" \
  --png outputs/media/score_piper_b1ns5.png --md outputs/piper/score_piper_b1ns5.md | tail -1
log "[3] b2 새 배치 평가"
( ev_t $POL piper_b2h "$EVQ" 0 1 2 3 4 5 6 7 8 9 ) & P1=$!; ( ev_p $POL piper_b2h "$EVQ" 8:0 8:3 8:5 8:7 8:9 4:5 ) & P2=$!; ( ev_p $POL piper_b2h "$EVQ" 4:9 1:7 8:1 8:4 1:8 4:1 ) & P3=$!
wait $P1 $P2 $P3
$PY scoreboard.py --forget outputs/piper_b2h_forget --switch outputs/piper_b2h_switch --name "블록 2차 piper_b2 (새 배치 20장면, 지연 11)" \
  --png outputs/media/score_piper_b2.png --md outputs/piper/score_piper_b2.md | tail -1
log "[4] b2 A2C2"
$PY a2c2.py gen --policy $POL --ttrtc --data $DATA --frac 0.5 --stride 16 --batch 8 --out data/a2c2_piper_b2 2>&1 | grep -E "끝|Traceback|Error"
$PY a2c2.py train --gen data/a2c2_piper_b2 --out outputs/a2c2_piper_b2 --steps 30000 --batch 64 --workers 4 2>&1 | grep -E "끝|val|Traceback|Error" | tail -3
A="--a2c2 outputs/a2c2_piper_b2"
( ev_t $POL piper_b2ah "$EVQ $A" 0 1 2 3 4 5 6 7 8 9 ) & P1=$!; ( ev_p $POL piper_b2ah "$EVQ $A" 8:0 8:3 8:5 8:7 8:9 4:5 ) & P2=$!; ( ev_p $POL piper_b2ah "$EVQ $A" 4:9 1:7 8:1 8:4 1:8 4:1 ) & P3=$!
wait $P1 $P2 $P3
$PY scoreboard.py --forget outputs/piper_b2ah_forget --switch outputs/piper_b2ah_switch --name "블록 2차 + A2C2 (새 배치 20장면, 지연 11)" \
  --png outputs/media/score_piper_b2a.png --md outputs/piper/score_piper_b2a.md | tail -1
log "끝"
