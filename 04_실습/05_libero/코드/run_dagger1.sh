#!/usr/bin/env bash
# DAgger 1회차 — 6차 모델이 실제로 가는 상태에서 전문가가 넘겨받아 마무리한 시범을 모아 6.1차로 이어서 학습
#
# 순서: 6차 평가 끝(run_v6.sh) → DAgger 수집 (원래 모델 평가와 동시에, GPU 3개 프로세스까지)
#       → 원래 모델·지연 평가까지 끝나면 → 6.1차 학습 (6차에서 이어서, 8000스텝) → 6.1차 평가
# ⚠️ 메모리: 학습은 다른 평가가 모두 끝난 뒤에만.
cd "$(dirname "$0")"
PY=.venv/bin/python
F='SUMMARY|STATS|Traceback|Error|  =='
log(){ echo "=== $(date +%H:%M:%S) $*"; }
busy(){ ps -eo comm,args | awk -v p="$1" '$1=="bash" && $0 ~ p && !/awk/' | grep -q .; }

log "6차 평가가 끝나기를 기다림"
while busy 'run_v6\.sh'; do sleep 120; done
[ -f outputs/v6_model/merged/model.safetensors ] || { log "6차 모델이 없다 — 멈춤"; exit 1; }

log "[1] DAgger 수집: 10개 태스크 + 전환 12조합, 장면 0~19·50~69, 넘겨받는 시점 무작위"
$PY dagger_v6.py --policy outputs/v6_model/merged --tasks 0 1 2 3 4 5 6 7 8 9 \
  --pairs 8:0,8:3,8:5,8:7,8:9,4:5,4:9,1:7,8:1,8:4,1:8,4:1 --episodes 0-19,50-69 --dart 0.1 \
  --out data/v6_dagger1 2>&1 | grep -E "$F"

log "원래 모델·지연 평가가 끝나기를 기다림 (메모리)"
while busy 'run_v6_base\.sh|run_v6_latency\.sh'; do sleep 120; done

log "[2] 6.1차 학습: 6차에서 이어서, 기존 시범 + DAgger, 8000스텝"
$PY train_lora.py --policy outputs/v6_model/merged --data data/v6_normal data/v6_switch data/v6_dagger1 \
  --full-expert --aug --balance --workers 0 --steps 8000 --batch-size 4 --grad-accum 2 --lr 2e-5 \
  --eval-every 2000 --save-every 4000 --log-every 200 --out outputs/v61_model 2>&1 | grep -E '에피소드 학습|val_loss|모델 저장|Traceback|Error'

POL=outputs/v61_model/merged
EV="--episodes 30 --start-episode 20"
( for t in 0 1 2 3 4 5 6 7 8 9; do $PY switch_experiment.py --policy $POL --task-a $t --strategy none $EV --out outputs/v61eval_forget 2>&1 | grep -E "$F"; done ) &
( for p in 8:0 8:3 8:5 8:7 8:9 4:5 4:9 1:7 8:1 8:4 1:8 4:1; do $PY switch_experiment.py --policy $POL --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy flush $EV --out outputs/v61eval_switch 2>&1 | grep -E "$F"; done ) &
wait
log "6.1차 평가 끝"
