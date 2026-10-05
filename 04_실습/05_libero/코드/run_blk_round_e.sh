#!/usr/bin/env bash
# 블록 라운드 E (10/4 밤에 미리 걸어 둠): 라운드 D 가 끝나면
#   ① b2 (또는 b2+A2C2 중 평균이 높은 쪽)이 95% 못 넘는 항목 교정 시범 (장면 2400~) ‖ 반복 5단계 시험 (지연 줄이기)
#   ② b3 = b2 + 교정 시범 두 묶음 + 4차 수집(빗나갔다 바로잡는 시범) → 28항목 ‖ A2C2 → b3 + A2C2 28항목
set -u
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 MUJOCO_GL=egl
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
until grep -q "=== .* 끝$" outputs/piper/blk_round_d.log 2>/dev/null; do sleep 120; done
B2=outputs/piper_b2_model/merged
EVQ="--latency-steps 11 --ttrtc --n-action-steps 1"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --frame-stride 2 --batch-size 4 --grad-accum 2 --log-every 500"
ev_t(){ local pol=$1 o=$2 opt=$3; shift 3; for t in "$@"; do $PY piper_sim/eval_piper.py --policy $pol --task-a $t --strategy none --episodes 20 --start-episode 1000 $opt --out outputs/${o}_forget 2>&1 | grep -E "$F"; done; }
ev_p(){ local pol=$1 o=$2 opt=$3; shift 3; for p in "$@"; do $PY piper_sim/eval_piper.py --policy $pol --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy keep --episodes 20 --start-episode 1000 $opt --out outputs/${o}_switch 2>&1 | grep -E "$F"; done; }
read -r _ M0 <<< "$($PY piper_sim/mean_score.py piper_b2h)"; read -r _ M1 <<< "$($PY piper_sim/mean_score.py piper_b2ah)"
if python3 -c "import sys; sys.exit(0 if $M1 > $M0 else 1)"; then BASE=piper_b2ah; DA="--a2c2 outputs/a2c2_piper_b2"; else BASE=piper_b2h; DA=""; fi
log "[1] 교정 시범: b2 평균 $M0, b2+A2C2 평균 $M1 → $BASE 의 95% 미만 항목"
eval "$($PY piper_sim/plan_dagger.py --base $BASE --start 2400)"
log "  단독 30:[${T30}] 20:[${T20}] 10:[${T10}] / 전환 30:[${P30}] 20:[${P20}] 10:[${P10}] / 재개 30:[${R30}] 20:[${R20}] 10:[${R10}]"
OUTD=data/piper_blk_dagger_b2
D="--policy $B2 $EVQ $DA --strategy keep --out $OUTD"
dag_t(){ for k in 30 20 10; do v=T$k; e=E$k; [ -n "${!v}" ] && $PY piper_sim/dagger_piper.py $D --tasks ${!v} --episodes ${!e} --seed $((51 + k)); done; }
dag_p(){ for k in 30 20 10; do v=P$k; e=E$k; [ -n "${!v}" ] && $PY piper_sim/dagger_piper.py $D --pairs ${!v} --episodes ${!e} --seed $((52 + k)); done; }
dag_r(){ for k in 30 20 10; do v=R$k; e=E$k; [ -n "${!v}" ] && $PY piper_sim/dagger_piper.py $D --pairs ${!v} --resume --episodes ${!e} --seed $((53 + k)); done; }
( dag_t > outputs/piper/dagger_b2_t.log 2>&1 ) & P1=$!
( dag_p > outputs/piper/dagger_b2_p.log 2>&1 ) & P2=$!
( dag_r > outputs/piper/dagger_b2_r.log 2>&1 ) & P3=$!
# 반복 5단계 (GPU 혼자 잰 생각 시간으로 지연) — 같은 항목 b2(h 또는 ah)와 비교
L5=$(python3 -c "import json,math; d=json.load(open('outputs/piper/bench_split_alone.json')); print(math.ceil(d['total_5steps_ms']/50))" 2>/dev/null || echo 8)
NS="--latency-steps $L5 --ttrtc --n-action-steps 1 --num-steps 5 $DA"
( ev_t $B2 piper_b2ns5 "$NS" 1 4 8 9; ev_p $B2 piper_b2ns5 "$NS" 8:1 8:5 4:9 ) & P4=$!
wait $P1 $P2 $P3 $P4
log "교정 시범 끝: $(ls $OUTD/episodes 2>/dev/null | wc -l). 반복 5단계(지연 $L5): $($PY piper_sim/mean_score.py piper_b2ns5)"
$PY piper_sim/trim_stalls.py $OUTD ${OUTD}_t
until grep -q "블록 4차 수집 끝" outputs/piper/blk_collect4.log 2>/dev/null; do sleep 120; done   # 빗나갔다 바로잡는 시범 (10/5)
DATA="data/piper_blk_normal_t data/piper_blk_switch_t data/piper_blk_normal2_t data/piper_blk_switch2_t data/piper_blk_normal3_t data/piper_blk_switch3_t data/piper_blk_normal4_t data/piper_blk_switch4_t data/piper_blk_dagger_b1_t ${OUTD}_t"
log "[2] b3 학습 (b2 에서)"
touch outputs/piper/TRAINING
$PY train_lora.py --policy $B2 --data $DATA $TR --dagger-frac 0.35 --steps 16000 --lr 3e-5 --eval-every 2000 --save-every 4000 \
  --out outputs/piper_b3_model 2>&1 | grep --line-buffered -E '에피소드 학습|val_loss|모델 저장|Traceback|Error'
rm -f outputs/piper/TRAINING
POL=outputs/piper_b3_model/merged
[ -f $POL/model.safetensors ] || { log "b3 모델 없음 — 멈춤"; exit 1; }
log "[3] b3 28항목 ‖ b3 A2C2"
( ev_t $POL piper_b3h "$EVQ" 0 1 2 3 4 5 6 7 8 9 ) & P1=$!; ( ev_p $POL piper_b3h "$EVQ" 8:0 8:3 8:5 8:7 8:9 4:5 ) & P2=$!; ( ev_p $POL piper_b3h "$EVQ" 4:9 1:7 8:1 8:4 1:8 4:1 ) & P3=$!
$PY a2c2.py gen --policy $POL --ttrtc --data $DATA --frac 0.5 --stride 16 --batch 8 --out data/a2c2_piper_b3 2>&1 | grep -E "끝|Traceback|Error"
$PY a2c2.py train --gen data/a2c2_piper_b3 --out outputs/a2c2_piper_b3 --steps 30000 --batch 64 --workers 4 2>&1 | grep -E "끝|val|Traceback|Error" | tail -3
wait $P1 $P2 $P3
$PY scoreboard.py --blocks --forget outputs/piper_b3h_forget --switch outputs/piper_b3h_switch --name "블록 3차 piper_b3 (새 배치 20장면, 지연 11)" \
  --png outputs/media/score_piper_b3.png --md outputs/piper/score_piper_b3.md | tail -1
$PY scoreboard.py --blocks --loose --forget outputs/piper_b3h_forget --switch outputs/piper_b3h_switch --name "블록 3차 piper_b3 (판 안 기준, 새 배치 20장면, 지연 11)" \
  --png outputs/media/score_piper_b3_loose.png --md outputs/piper/score_piper_b3_loose.md | tail -1
log "[4] b3 + A2C2"
A="--a2c2 outputs/a2c2_piper_b3"
( ev_t $POL piper_b3ah "$EVQ $A" 0 1 2 3 4 5 6 7 8 9 ) & P1=$!; ( ev_p $POL piper_b3ah "$EVQ $A" 8:0 8:3 8:5 8:7 8:9 4:5 ) & P2=$!; ( ev_p $POL piper_b3ah "$EVQ $A" 4:9 1:7 8:1 8:4 1:8 4:1 ) & P3=$!
wait $P1 $P2 $P3
$PY scoreboard.py --blocks --forget outputs/piper_b3ah_forget --switch outputs/piper_b3ah_switch --name "블록 3차 + A2C2 (새 배치 20장면, 지연 11)" \
  --png outputs/media/score_piper_b3a.png --md outputs/piper/score_piper_b3a.md | tail -1
$PY scoreboard.py --blocks --loose --forget outputs/piper_b3ah_forget --switch outputs/piper_b3ah_switch --name "블록 3차 + A2C2 (판 안 기준, 새 배치 20장면, 지연 11)" \
  --png outputs/media/score_piper_b3a_loose.png --md outputs/piper/score_piper_b3a_loose.md | tail -1
log "끝"
