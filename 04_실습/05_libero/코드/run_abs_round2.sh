#!/usr/bin/env bash
# 절대 목표 2차 (10/5 밤에 미리 걸어 둠): run_abs_round.sh 가 끝나면
#   ① ba1 이 95% 못 넘는 항목 교정 시범 (실제 배치 조건, 장면 2500~, 3프로세스)
#   ② ba2 = ba1 + 교정 시범 (절대 목표 그대로) → 28항목 (3cm·판 안)
set -u
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 MUJOCO_GL=egl
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
until grep -q "=== .* 끝$" outputs/piper/abs_round.log 2>/dev/null; do sleep 120; done
B=outputs/piper_ba1_model/merged
EVQ="--latency-steps 11 --ttrtc --n-action-steps 1"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --frame-stride 2 --batch-size 4 --grad-accum 2 --log-every 500"
DATA="data/piper_blk_normal_t data/piper_blk_switch_t data/piper_blk_normal2_t data/piper_blk_switch2_t data/piper_blk_normal3_t data/piper_blk_switch3_t data/piper_blk_normal4_t data/piper_blk_switch4_t data/piper_blk_dagger_b1_t"
ev_t(){ local pol=$1 o=$2 opt=$3; shift 3; for t in "$@"; do $PY piper_sim/eval_piper.py --policy $pol --task-a $t --strategy none --episodes 20 --start-episode 1000 $opt --out outputs/${o}_forget 2>&1 | grep -E "$F"; done; }
ev_p(){ local pol=$1 o=$2 opt=$3; shift 3; for p in "$@"; do $PY piper_sim/eval_piper.py --policy $pol --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy keep --episodes 20 --start-episode 1000 $opt --out outputs/${o}_switch 2>&1 | grep -E "$F"; done; }
board(){ $PY scoreboard.py --blocks --forget outputs/$1_forget --switch outputs/$1_switch --name "$2" --png outputs/media/score_$1.png --md outputs/piper/score_$1.md | tail -1
         $PY scoreboard.py --blocks --loose --forget outputs/$1_forget --switch outputs/$1_switch --name "$2 판 안 기준" --png outputs/media/score_$1_loose.png --md outputs/piper/score_$1_loose.md | tail -1; }
eval "$($PY piper_sim/plan_dagger.py --base piper_ba1h --start 2500)"
log "[1] ba1 교정 시범 — 단독 30:[${T30}] 20:[${T20}] 10:[${T10}] / 전환 30:[${P30}] 20:[${P20}] 10:[${P10}] / 재개 30:[${R30}] 20:[${R20}] 10:[${R10}]"
OUTD=data/piper_blk_dagger_ba1
D="--policy $B $EVQ --strategy keep --out $OUTD"
dag_t(){ for k in 30 20 10; do v=T$k; e=E$k; [ -n "${!v}" ] && $PY piper_sim/dagger_piper.py $D --tasks ${!v} --episodes ${!e} --seed $((61 + k)); done; }
dag_p(){ for k in 30 20 10; do v=P$k; e=E$k; [ -n "${!v}" ] && $PY piper_sim/dagger_piper.py $D --pairs ${!v} --episodes ${!e} --seed $((62 + k)); done; }
dag_r(){ for k in 30 20 10; do v=R$k; e=E$k; [ -n "${!v}" ] && $PY piper_sim/dagger_piper.py $D --pairs ${!v} --resume --episodes ${!e} --seed $((63 + k)); done; }
( dag_t > outputs/piper/dagger_ba1_t.log 2>&1 ) & P1=$!
( dag_p > outputs/piper/dagger_ba1_p.log 2>&1 ) & P2=$!
( dag_r > outputs/piper/dagger_ba1_r.log 2>&1 ) & P3=$!
wait $P1 $P2 $P3
log "교정 시범 끝: $(ls $OUTD/episodes 2>/dev/null | wc -l) — $(grep -h STATS outputs/piper/dagger_ba1_*.log | tr '\n' ' ')"
$PY piper_sim/trim_stalls.py $OUTD ${OUTD}_t
until grep -q "회복 시범 수집 끝" outputs/piper/blk_recovery.log 2>/dev/null; do sleep 120; done
until grep -q "넓은 배치 수집 끝" outputs/piper/blk_collect_wide.log 2>/dev/null; do sleep 120; done   # 10/6: 물체를 보고 가게
log "[2] ba2 학습 (ba1 에서, 절대 목표, 20000스텝, + 넓은 배치 + 회복 시범 $(ls data/piper_blk_recovery_t/episodes | wc -l))"
touch outputs/piper/TRAINING
$PY train_lora.py --policy $B --data $DATA ${OUTD}_t data/piper_blk_recovery_t data/piper_blk_wide_normal_t data/piper_blk_wide_switch_t $TR --abs-pos --dagger-frac 0.3 --steps 20000 --lr 4e-5 --eval-every 2000 --save-every 4000 \
  --out outputs/piper_ba2_model 2>&1 | grep --line-buffered -E '에피소드 학습|val_loss|모델 저장|Traceback|Error'
rm -f outputs/piper/TRAINING
POL=outputs/piper_ba2_model/merged
[ -f $POL/abs_pos ] || { log "ba2 모델 없음 — 멈춤"; exit 1; }
log "[3] ba2 28항목"
( ev_t $POL piper_ba2h "$EVQ" 0 1 2 3 4 5 6 7 8 9 ) & P1=$!; ( ev_p $POL piper_ba2h "$EVQ" 8:0 8:3 8:5 8:7 8:9 4:5 ) & P2=$!; ( ev_p $POL piper_ba2h "$EVQ" 4:9 1:7 8:1 8:4 1:8 4:1 ) & P3=$!
wait $P1 $P2 $P3
board piper_ba2h "블록 절대 목표 ba2 (새 배치 20장면, 지연 11)"
log "끝"
