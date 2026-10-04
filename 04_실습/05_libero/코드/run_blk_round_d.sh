#!/usr/bin/env bash
# 블록 라운드 D (10/4 23시): 라운드 C 를 이어받는다 (b1c 학습·b1 교정 시범은 C 가 띄운 것이 계속 돈다).
#   바뀐 것: b2 학습 데이터에 3차 수집(내려가는 속도·멈춤을 다르게 한 시범)을 더하고, b2 학습 동안 b1c 28항목을 잰다.
set -u
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 MUJOCO_GL=egl
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
B1=outputs/piper_b1_model/merged; B1C=outputs/piper_b1c_model/merged
EVQ="--latency-steps 11 --ttrtc --n-action-steps 1"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --frame-stride 2 --batch-size 4 --grad-accum 2 --log-every 500"
OUTD=data/piper_blk_dagger_b1
ev_t(){ local pol=$1 o=$2 opt=$3; shift 3; for t in "$@"; do $PY piper_sim/eval_piper.py --policy $pol --task-a $t --strategy none --episodes 20 --start-episode 1000 $opt --out outputs/${o}_forget 2>&1 | grep -E "$F"; done; }
ev_p(){ local pol=$1 o=$2 opt=$3; shift 3; for p in "$@"; do $PY piper_sim/eval_piper.py --policy $pol --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy keep --episodes 20 --start-episode 1000 $opt --out outputs/${o}_switch 2>&1 | grep -E "$F"; done; }
while pgrep -f "dagger_piper[.]py .*piper_blk_dagger_b1" >/dev/null; do sleep 60; done
log "교정 시범 끝: $(ls $OUTD/episodes 2>/dev/null | wc -l) — $(grep -h STATS outputs/piper/dagger_b1_*.log | tr '\n' ' ')"
$PY piper_sim/trim_stalls.py $OUTD ${OUTD}_t
log "[2] b1 (고친 평가) 일부 항목 — 비교 기준"
( ev_t $B1 piper_b1fx "$EVQ" 1 4 8 9 ) & P1=$!; ( ev_p $B1 piper_b1fx "$EVQ" 8:1 8:5 4:9 ) & P2=$!
wait $P1 $P2
while pgrep -f "train_lora[.]py .*piper_b1c_model" >/dev/null; do sleep 60; done
rm -f outputs/piper/TRAINING
[ -f $B1C/model.safetensors ] || { log "b1c 모델 없음 — 멈춤"; exit 1; }
log "[3] b1c 같은 항목"
( ev_t $B1C piper_b1cx "$EVQ" 1 4 8 9 ) & P1=$!; ( ev_p $B1C piper_b1cx "$EVQ" 8:1 8:5 4:9 ) & P2=$!
wait $P1 $P2
$PY piper_sim/mean_score.py piper_b1fx piper_b1cx | while read l; do log "  평균 $l"; done
until grep -q "블록 3차 수집 끝" outputs/piper/blk_collect3.log 2>/dev/null; do sleep 60; done
DATA="data/piper_blk_normal_t data/piper_blk_switch_t data/piper_blk_normal2_t data/piper_blk_switch2_t data/piper_blk_normal3_t data/piper_blk_switch3_t"
log "[4] b2 학습 (b1c 에서, + 3차 수집 + 교정 시범, 20000스텝) ‖ b1c 28항목"
touch outputs/piper/TRAINING
( $PY train_lora.py --policy $B1C --data $DATA ${OUTD}_t $TR --dagger-frac 0.3 --steps 20000 --lr 3e-5 --eval-every 2500 --save-every 4000 \
    --out outputs/piper_b2_model 2>&1 | grep --line-buffered -E '에피소드 학습|val_loss|모델 저장|Traceback|Error' > outputs/piper/train_b2.log ) & PT=$!
( ev_t $B1C piper_b1ch "$EVQ" 0 1 2 3 4 5 6 7 8 9 ) & P1=$!; ( ev_p $B1C piper_b1ch "$EVQ" 8:0 8:3 8:5 8:7 8:9 4:5 ) & P2=$!; ( ev_p $B1C piper_b1ch "$EVQ" 4:9 1:7 8:1 8:4 1:8 4:1 ) & P3=$!
wait $P1 $P2 $P3
$PY scoreboard.py --forget outputs/piper_b1ch_forget --switch outputs/piper_b1ch_switch --name "블록 b1c (새 배치 20장면, 지연 11)" \
  --png outputs/media/score_piper_b1c.png --md outputs/piper/score_piper_b1c.md | tail -1
wait $PT
rm -f outputs/piper/TRAINING
POL=outputs/piper_b2_model/merged
[ -f $POL/model.safetensors ] || { log "b2 모델 없음 — 멈춤"; exit 1; }
log "[5] b2 새 배치 28항목 ‖ b2 A2C2"
( ev_t $POL piper_b2h "$EVQ" 0 1 2 3 4 5 6 7 8 9 ) & P1=$!; ( ev_p $POL piper_b2h "$EVQ" 8:0 8:3 8:5 8:7 8:9 4:5 ) & P2=$!; ( ev_p $POL piper_b2h "$EVQ" 4:9 1:7 8:1 8:4 1:8 4:1 ) & P3=$!
$PY a2c2.py gen --policy $POL --ttrtc --data $DATA ${OUTD}_t --frac 0.5 --stride 16 --batch 8 --out data/a2c2_piper_b2 2>&1 | grep -E "끝|Traceback|Error"
$PY a2c2.py train --gen data/a2c2_piper_b2 --out outputs/a2c2_piper_b2 --steps 30000 --batch 64 --workers 4 2>&1 | grep -E "끝|val|Traceback|Error" | tail -3
wait $P1 $P2 $P3
$PY scoreboard.py --forget outputs/piper_b2h_forget --switch outputs/piper_b2h_switch --name "블록 2차 piper_b2 (새 배치 20장면, 지연 11)" \
  --png outputs/media/score_piper_b2.png --md outputs/piper/score_piper_b2.md | tail -1
log "[6] b2 + A2C2"
A="--a2c2 outputs/a2c2_piper_b2"
( ev_t $POL piper_b2ah "$EVQ $A" 0 1 2 3 4 5 6 7 8 9 ) & P1=$!; ( ev_p $POL piper_b2ah "$EVQ $A" 8:0 8:3 8:5 8:7 8:9 4:5 ) & P2=$!; ( ev_p $POL piper_b2ah "$EVQ $A" 4:9 1:7 8:1 8:4 1:8 4:1 ) & P3=$!
wait $P1 $P2 $P3
$PY scoreboard.py --forget outputs/piper_b2ah_forget --switch outputs/piper_b2ah_switch --name "블록 2차 + A2C2 (새 배치 20장면, 지연 11)" \
  --png outputs/media/score_piper_b2a.png --md outputs/piper/score_piper_b2a.md | tail -1
log "끝"
