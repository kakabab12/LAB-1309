#!/usr/bin/env bash
# 블록 라운드 D2 (10/5 11:40): D 가 대기 조건 실수로 멈춰 있던 것을 이어서.
#   ⚠️ pgrep -f 패턴이 다른 대기용 셸의 명령줄(스크립트 본문)에도 걸려 9시간을 기다렸다 → 대기는 PID 로만.
#   b2 학습(b1c 에서) ‖ b1·b1c 일부 항목 평가(첫 확인 숫자) → b2 28항목 ‖ A2C2 → b2+A2C2 28항목. 끝나면 라운드 E 가 이어받는다.
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
log "b1 교정 시범: $(ls $OUTD/episodes | wc -l) — $(grep -h STATS outputs/piper/dagger_b1_*.log | tr '\n' ' ')"
[ -d ${OUTD}_t ] || $PY piper_sim/trim_stalls.py $OUTD ${OUTD}_t
DATA="data/piper_blk_normal_t data/piper_blk_switch_t data/piper_blk_normal2_t data/piper_blk_switch2_t data/piper_blk_normal3_t data/piper_blk_switch3_t"
log "[1] b2 학습 (b1c 에서, + 3차 수집 + 교정 시범, 20000스텝) ‖ b1c 일부 항목(2) ‖ b1 일부 항목(1)"
touch outputs/piper/TRAINING
( $PY train_lora.py --policy $B1C --data $DATA ${OUTD}_t $TR --dagger-frac 0.3 --steps 20000 --lr 3e-5 --eval-every 2500 --save-every 4000 \
    --out outputs/piper_b2_model 2>&1 | grep --line-buffered -E '에피소드 학습|val_loss|모델 저장|Traceback|Error' > outputs/piper/train_b2.log ) & PT=$!
( ev_t $B1C piper_b1cx "$EVQ" 8 1 4 9 ) & P1=$!; ( ev_p $B1C piper_b1cx "$EVQ" 8:1 8:5 4:9 ) & P2=$!
( ev_t $B1 piper_b1fx "$EVQ" 8 1 4 9; ev_p $B1 piper_b1fx "$EVQ" 8:1 8:5 4:9 ) & P3=$!
wait $P1 $P2
log "  b1c 일부 항목 끝: $($PY piper_sim/mean_score.py piper_b1cx)"
wait $P3
log "  b1 일부 항목 끝: $($PY piper_sim/mean_score.py piper_b1fx)"
wait $PT
rm -f outputs/piper/TRAINING
POL=outputs/piper_b2_model/merged
[ -f $POL/model.safetensors ] || { log "b2 모델 없음 — 멈춤"; exit 1; }
log "[2] b2 새 배치 28항목 ‖ b2 A2C2"
( ev_t $POL piper_b2h "$EVQ" 0 1 2 3 4 5 6 7 8 9 ) & P1=$!; ( ev_p $POL piper_b2h "$EVQ" 8:0 8:3 8:5 8:7 8:9 4:5 ) & P2=$!; ( ev_p $POL piper_b2h "$EVQ" 4:9 1:7 8:1 8:4 1:8 4:1 ) & P3=$!
$PY a2c2.py gen --policy $POL --ttrtc --data $DATA ${OUTD}_t --frac 0.5 --stride 16 --batch 8 --out data/a2c2_piper_b2 2>&1 | grep -E "끝|Traceback|Error"
$PY a2c2.py train --gen data/a2c2_piper_b2 --out outputs/a2c2_piper_b2 --steps 30000 --batch 64 --workers 4 2>&1 | grep -E "끝|val|Traceback|Error" | tail -3
wait $P1 $P2 $P3
$PY scoreboard.py --forget outputs/piper_b2h_forget --switch outputs/piper_b2h_switch --name "블록 2차 piper_b2 (새 배치 20장면, 지연 11)" \
  --png outputs/media/score_piper_b2.png --md outputs/piper/score_piper_b2.md | tail -1
log "[3] b2 + A2C2"
A="--a2c2 outputs/a2c2_piper_b2"
( ev_t $POL piper_b2ah "$EVQ $A" 0 1 2 3 4 5 6 7 8 9 ) & P1=$!; ( ev_p $POL piper_b2ah "$EVQ $A" 8:0 8:3 8:5 8:7 8:9 4:5 ) & P2=$!; ( ev_p $POL piper_b2ah "$EVQ $A" 4:9 1:7 8:1 8:4 1:8 4:1 ) & P3=$!
wait $P1 $P2 $P3
$PY scoreboard.py --forget outputs/piper_b2ah_forget --switch outputs/piper_b2ah_switch --name "블록 2차 + A2C2 (새 배치 20장면, 지연 11)" \
  --png outputs/media/score_piper_b2a.png --md outputs/piper/score_piper_b2a.md | tail -1
log "끝"
