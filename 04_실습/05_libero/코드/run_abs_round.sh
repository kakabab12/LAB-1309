#!/usr/bin/env bash
# 절대 목표 위치 라운드 (10/5 21시): 시범을 눈 감고 다시 실행하면 변화량 8/12(최대 오차 4.7cm) vs 절대 목표 12/12(1.4cm)
#   ① b2(변화량) 학습이 끝나면 → ba1 = b2 에서 절대 목표 위치로 학습 (--abs-pos --renorm-action) ‖ b2 28항목(기록·비교)
#   ② ba1 28항목 (3cm·판 안) → ③ ba1 교정 시범 → ba2 → 28항목
set -u
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 MUJOCO_GL=egl
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 11 --ttrtc --n-action-steps 1"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --frame-stride 2 --batch-size 4 --grad-accum 2 --log-every 500"
DATA="data/piper_blk_normal_t data/piper_blk_switch_t data/piper_blk_normal2_t data/piper_blk_switch2_t data/piper_blk_normal3_t data/piper_blk_switch3_t data/piper_blk_normal4_t data/piper_blk_switch4_t"
ev_t(){ local pol=$1 o=$2 opt=$3; shift 3; for t in "$@"; do $PY piper_sim/eval_piper.py --policy $pol --task-a $t --strategy none --episodes 20 --start-episode 1000 $opt --out outputs/${o}_forget 2>&1 | grep -E "$F"; done; }
ev_p(){ local pol=$1 o=$2 opt=$3; shift 3; for p in "$@"; do $PY piper_sim/eval_piper.py --policy $pol --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy keep --episodes 20 --start-episode 1000 $opt --out outputs/${o}_switch 2>&1 | grep -E "$F"; done; }
board(){ $PY scoreboard.py --blocks --forget outputs/$1_forget --switch outputs/$1_switch --name "$2" --png outputs/media/score_$1.png --md outputs/piper/score_$1.md | tail -1
         $PY scoreboard.py --blocks --loose --forget outputs/$1_forget --switch outputs/$1_switch --name "$2 판 안 기준" --png outputs/media/score_$1_loose.png --md outputs/piper/score_$1_loose.md | tail -1; }
while kill -0 1919886 2>/dev/null; do sleep 30; done
B2=outputs/piper_b2_model/merged
[ -f $B2/model.safetensors ] || { log "b2 모델 없음 — 멈춤"; exit 1; }
log "[1] ba1 학습 (b2 에서, 절대 목표 위치, 24000스텝) ‖ b2 28항목"
touch outputs/piper/TRAINING
( $PY train_lora.py --policy $B2 --data $DATA data/piper_blk_dagger_b1_t $TR --abs-pos --renorm-action --dagger-frac 0.2 --steps 24000 --lr 5e-5 \
    --eval-every 3000 --save-every 4000 --out outputs/piper_ba1_model 2>&1 | grep --line-buffered -E '에피소드 학습|val_loss|모델 저장|Traceback|Error' > outputs/piper/train_ba1.log ) & PT=$!
( ev_t $B2 piper_b2h "$EVQ" 0 1 2 3 4 5 6 7 8 9 ) & P1=$!; ( ev_p $B2 piper_b2h "$EVQ" 8:0 8:3 8:5 8:7 8:9 4:5 ) & P2=$!; ( ev_p $B2 piper_b2h "$EVQ" 4:9 1:7 8:1 8:4 1:8 4:1 ) & P3=$!
wait $P1 $P2 $P3
board piper_b2h "블록 2차 b2 (변화량, 새 배치 20장면, 지연 11)"
wait $PT
rm -f outputs/piper/TRAINING
POL=outputs/piper_ba1_model/merged
[ -f $POL/abs_pos ] || { log "ba1 모델(절대 목표 표시) 없음 — 멈춤"; exit 1; }
log "[2] ba1 28항목"
( ev_t $POL piper_ba1h "$EVQ" 0 1 2 3 4 5 6 7 8 9 ) & P1=$!; ( ev_p $POL piper_ba1h "$EVQ" 8:0 8:3 8:5 8:7 8:9 4:5 ) & P2=$!; ( ev_p $POL piper_ba1h "$EVQ" 4:9 1:7 8:1 8:4 1:8 4:1 ) & P3=$!
wait $P1 $P2 $P3
board piper_ba1h "블록 절대 목표 ba1 (새 배치 20장면, 지연 11)"
log "끝"
