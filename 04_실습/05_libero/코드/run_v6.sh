#!/usr/bin/env bash
# 6차 — 전부 전문가 방식의 시범으로 동작 전문가 전체 미세조정 → 평가 (2026-10-01)
#
# 5차 실패(75% → 39%)의 원인: 스크립트 시범과 원래 모델 동작이 같은 지시문에 섞임.
# 6차는 정상·전환·재개를 모두 같은 전문가(10개 태스크 98.7%)로 만들었다.
#
# ⚠️ 메모리: 이 PC 는 31GB, 스왑 0. 수집(5개 × 2.4GB)이 끝난 뒤에만 학습한다.
#    학습은 데이터 로더 작업자 0 (작업자마다 6GB 데이터가 복사되는 것을 막는다).
#    평가는 동시에 2개까지 (각 약 4GB).
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
F='SUMMARY|STATS|Traceback|Error|  =='
log(){ echo "=== $(date +%H:%M:%S) $*"; }

log "수집이 끝나기를 기다림"
while ps -eo comm,args | awk '$1=="python" && /collect_v6\.py/' | grep -q .; do sleep 60; done
log "수집 끝: 정상 $(ls data/v6_normal/episodes | wc -l), 전환 $(ls data/v6_switch/episodes | grep -c npz)"

log "[학습] 동작 전문가 전체 미세조정, 이미지 증강, 24000 스텝"
$PY train_lora.py --data data/v6_normal data/v6_switch --full-expert --aug --balance --workers 0 \
  --steps 24000 --batch-size 4 --grad-accum 2 --lr 5e-5 --eval-every 2000 --save-every 4000 --log-every 200 \
  --out outputs/v6_model 2>&1 | grep -E '전체 미세조정|에피소드 학습|val_loss|모델 저장|Traceback|Error'

POL=outputs/v6_model/merged
EV="--episodes 30 --start-episode 20"
(
  log "[평가 1] 10개 태스크 단독, 장면 20~49"
  for t in 0 1 2 3 4 5 6 7 8 9; do
    $PY switch_experiment.py --policy "$POL" --task-a "$t" --strategy none $EV --out outputs/v6eval_forget 2>&1 | grep -E "$F"
  done
) &
(
  log "[평가 2] 전환 (실패하던 쌍 + 잘 되던 쌍), 장면 20~49"
  for p in 8:0 8:3 8:5 8:7 8:9 4:5 4:9 1:7 8:1 8:4 1:8 4:1; do
    $PY switch_experiment.py --policy "$POL" --task-a "${p%:*}" --task-b "${p#*:}" --switch-at grasp:3 \
      --strategy flush $EV --out outputs/v6eval_switch 2>&1 | grep -E "$F"
  done
) &
wait
log "끝"
