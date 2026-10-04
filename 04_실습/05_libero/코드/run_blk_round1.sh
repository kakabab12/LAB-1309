#!/usr/bin/env bash
# 블록 장면 학생 모델 1차 piper_b1 (10/4): 블록 시범으로 학습 → 쉬운 점검 → 새 배치 28항목 → A2C2(매 스텝 보정) → 다시 평가
#   출발: PiPER 2차(원래 물체, 깊게 쥐기) 학습의 마지막 저장본 — PiPER 팔 움직임을 이미 배운 모델.
#   블록 장면이 본 실험이 되어 원래 물체 2차 학습은 여기서 멈춘다.
set -u
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 MUJOCO_GL=egl
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
until grep -q "블록 수집 끝" outputs/piper/blk_collect.log 2>/dev/null; do sleep 60; done
pkill -f "train_lora[.]py .*piper_s2_model" && sleep 20
INIT=$(ls -d outputs/piper_s2_model/step_* | sort -t_ -k4 -n | tail -1)
DATA="data/piper_blk_normal_t data/piper_blk_switch_t"
log "[1] piper_b1 학습 ($INIT 에서): 정상 $(ls data/piper_blk_normal_t/episodes | wc -l), 전환·재개 $(ls data/piper_blk_switch_t/episodes | wc -l)"
touch outputs/piper/TRAINING
$PY train_lora.py --policy $INIT --data $DATA --full-expert --aug --balance --workers 0 --rtc-max-delay 14 \
  --ema 0.999 --frame-stride 2 --steps 20000 --batch-size 4 --grad-accum 2 --lr 5e-5 --eval-every 2500 --save-every 4000 \
  --log-every 500 --out outputs/piper_b1_model 2>&1 | grep --line-buffered -E '에피소드 학습|val_loss|모델 저장|Traceback|Error'
rm -f outputs/piper/TRAINING
POL=outputs/piper_b1_model/merged
[ -f $POL/model.safetensors ] || { log "모델 없음 — 멈춤"; exit 1; }
EVQ="--latency-steps 11 --ttrtc --n-action-steps 1"
log "[2] 쉬운 점검: T8 학습 배치 2000~2004, 지연 11"
$PY piper_sim/eval_piper.py --policy $POL --task-a 8 --strategy none --episodes 5 --start-episode 2000 $EVQ --out outputs/piper_b1_sanity 2>&1 | grep -E "$F"
EV="--episodes 20 --start-episode 1000 $EVQ"
run_t(){ local o=$1; shift; for t in "$@"; do $PY piper_sim/eval_piper.py --policy $POL --task-a $t --strategy none $EV $AOPT --out outputs/${o}_forget 2>&1 | grep -E "$F"; done; }
run_p(){ local o=$1; shift; for p in "$@"; do $PY piper_sim/eval_piper.py --policy $POL --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy keep $EV $AOPT --out outputs/${o}_switch 2>&1 | grep -E "$F"; done; }
log "[3] 새 배치 1000~1019 평가 + A2C2 데이터"
AOPT=""
( run_t piper_b1h 0 1 2 3 4 5 6 7 8 9 ) & ( run_p piper_b1h 8:0 8:3 8:5 8:7 8:9 4:5 ) & ( run_p piper_b1h 4:9 1:7 8:1 8:4 1:8 4:1 ) &
( $PY a2c2.py gen --policy $POL --ttrtc --data $DATA --frac 0.5 --stride 16 --batch 8 --out data/a2c2_piper_b1 2>&1 | grep -E "끝|Traceback|Error" ) &
wait
$PY scoreboard.py --forget outputs/piper_b1h_forget --switch outputs/piper_b1h_switch --name "블록 1차 piper_b1 (새 배치 20장면, 지연 11)" \
  --png outputs/media/score_piper_b1.png --md outputs/piper/score_piper_b1.md | tail -1
log "[4] A2C2 학습"
$PY a2c2.py train --gen data/a2c2_piper_b1 --out outputs/a2c2_piper_b1 --steps 30000 --batch 64 --workers 4 2>&1 | grep -E "끝|val|Traceback|Error" | tail -3
log "[5] piper_b1 + A2C2 새 배치 평가"
AOPT="--a2c2 outputs/a2c2_piper_b1"
( run_t piper_b1ah 0 1 2 3 4 ) & ( run_t piper_b1ah 5 6 7 8 9 ) & ( run_p piper_b1ah 8:0 8:3 8:5 8:7 8:9 4:5 ) & ( run_p piper_b1ah 4:9 1:7 8:1 8:4 1:8 4:1 ) &
wait
$PY scoreboard.py --forget outputs/piper_b1ah_forget --switch outputs/piper_b1ah_switch --name "블록 1차 + A2C2 (새 배치 20장면, 지연 11)" \
  --png outputs/media/score_piper_b1a.png --md outputs/piper/score_piper_b1a.md | tail -1
log "끝"
