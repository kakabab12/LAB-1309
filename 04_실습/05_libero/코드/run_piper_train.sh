#!/usr/bin/env bash
# PiPER 학생 모델 1차 (2026-10-03): PiPER 시범으로 Panda 모델(v6b)에서 이어 학습 → 점검 → 새 배치 평가
#   학습 설정은 v6c3 와 같다 (지연 흉내, EMA, 프레임 2개마다 1개, 동작 정규화는 그대로)
cd "$(dirname "$0")"
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
until grep -q "PiPER 수집 끝" outputs/piper/run_collect.log 2>/dev/null; do sleep 60; done
while ps -eo comm,args | awk '$1=="python" && /train_lora\.py/' | grep -q .; do sleep 60; done   # Panda 학습이 끝날 때까지
touch outputs/piper/TRAINING
trap 'rm -f outputs/piper/TRAINING' EXIT
log "PiPER 시범 (오래 멈춘 구간 잘라 냄): 정상 $(ls data/piper_normal_t/episodes | wc -l), 전환 $(ls data/piper_switch_t/episodes | wc -l)"
log "[1] PiPER 학생 1차 학습 (v6b 에서)"
$PY train_lora.py --policy outputs/v6b_model/merged --data data/piper_normal_t data/piper_switch_t \
  --full-expert --aug --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --frame-stride 2 \
  --steps 20000 --batch-size 4 --grad-accum 2 --lr 5e-5 --eval-every 2500 --save-every 5000 --log-every 500 \
  --out outputs/piper_s1_model 2>&1 | grep --line-buffered -E '지연 흉내|에피소드 학습|val_loss|모델 저장|Traceback|Error'
rm -f outputs/piper/TRAINING
POL=outputs/piper_s1_model/merged
[ -f $POL/model.safetensors ] || { log "모델 없음 — 멈춤"; exit 1; }
# 쉬운 점검도 실제 배치 조건으로 (지연 11 + 앞부분 제공). 지연 흉내내기로 학습한 모델을 앞부분 없이 재면
#   Panda v6c3 가 0/5 였지만 배치 조건에서는 2/5 — 점검 조건이 학습과 달랐다 (10/3)
log "[2] 쉬운 점검: T8 학습 배치(2000~2004), 지연 11, 계속 계산"
$PY piper_sim/eval_piper.py --policy $POL --task-a 8 --strategy none --episodes 5 --start-episode 2000 --latency-steps 11 --ttrtc --n-action-steps 1 --out outputs/piper_s1_sanity 2>&1 | grep -E "$F"
log "[3] 새 배치 1000~1019, 지연 11스텝, 계속 계산"
EV="--episodes 20 --start-episode 1000 --latency-steps 11 --ttrtc --n-action-steps 1"
run_t(){ for t in "$@"; do $PY piper_sim/eval_piper.py --policy $POL --task-a $t --strategy none $EV --out outputs/piper_s1h_forget 2>&1 | grep -E "$F"; done; }
run_p(){ for p in "$@"; do $PY piper_sim/eval_piper.py --policy $POL --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy keep $EV --out outputs/piper_s1h_switch 2>&1 | grep -E "$F"; done; }
( run_t 0 1 2 3 4 ) & ( run_t 5 6 7 8 9 ) & ( run_p 8:0 8:3 8:5 8:7 8:9 4:5 ) & ( run_p 4:9 1:7 8:1 8:4 1:8 4:1 ) &
wait
$PY scoreboard.py --forget outputs/piper_s1h_forget --switch outputs/piper_s1h_switch --name "PiPER 1차 (새 배치 20장면, 지연 11)" \
  --png outputs/media/score_piper_s1.png --md outputs/piper/score_piper_s1.md | tail -1
log "끝"
