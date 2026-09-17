#!/usr/bin/env bash
# v2: 1차 LoRA 실패 원인(데이터 편향 → 못하는 쌍은 데이터 0개, 잘하던 태스크를 잊음)을 고친 재학습
#  (1) 교란 없는 정상 궤적 — 원래 잘하던 것을 잊지 않게
#  (2) 먼 위치 교란 8~20cm — ablation 에서 전환 시 실제 위치 이탈이 12~27cm 로 측정됨
#  (3) hindsight 전환 데이터 — B 에 실패해도 우연히 달성한 다른 태스크로 라벨을 바꿔 저장
#  학습: 태스크별 균형 샘플링, LoRA r=8, lr 5e-5 (망각 완화)
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
log(){ echo "=== $(date +%H:%M:%S) $*"; }
OUT=data/v2

log "1/5 정상 궤적 (교란 없음, 10개 태스크)"
for t in 0 1 2 3 4 5 6 7 8 9; do echo "--tasks $t"; done | xargs -P 3 -I{} sh -c \
  "$PY collect_robust_data.py {} --mode pose --episodes 15 --start-episode 200 --max-offset-cm 0 --max-yaw-deg 0 --out $OUT 2>&1 | grep -E '^STATS|Traceback'"

log "2/5 먼 위치 교란 (8~20cm)"
for t in 0 1 2 4 5 7 8; do echo "--tasks $t"; done | xargs -P 3 -I{} sh -c \
  "$PY collect_robust_data.py {} --mode pose --episodes 30 --start-episode 100 --min-offset-cm 8 --max-offset-cm 20 --max-yaw-deg 10 --out $OUT 2>&1 | grep -E '^STATS|Traceback'"

log "3/5 hindsight 전환 데이터"
for p in 8:7 4:7 1:5 2:5; do for k in grasp:3 grasp:20; do
  echo "--tasks ${p%:*} --task-b ${p#*:} --switch-at $k"; done; done | xargs -P 3 -I{} sh -c \
  "$PY collect_robust_data.py {} --mode hindsight --episodes 25 --start-episode 100 --out $OUT 2>&1 | grep -E '^STATS|Traceback'"
echo "v2 에피소드: $(ls $OUT/episodes | wc -l) (S=전환 성공, H=hindsight, T=정상/교란)"
ls $OUT/episodes | cut -c1 | sort | uniq -c

log "4/5 LoRA 재학습 (균형 샘플링)"
$PY train_lora.py --data data/robust $OUT --balance --steps 2500 --batch-size 4 --grad-accum 2 \
  --lr 5e-5 --lora-r 8 --lora-alpha 16 --out outputs/lora_v2 2>&1 | grep -E 'LoRA|에피소드|태스크별|val_loss|합친|Traceback|Error'

M=outputs/lora_v2/merged
log "5/5 평가"
{
  for t in 7 1 5; do
    for o in 0 4 8; do echo "pose --task $t --offset-cm $o"; done
    for y in 10 20 30; do echo "pose --task $t --yaw-deg $y"; done
  done
  for p in 8:7 4:7 1:5 2:5; do for k in grasp:3 grasp:20; do
    echo "switch --task-a ${p%:*} --task-b ${p#*:} --switch-at $k"; done; done
} | xargs -P 3 -I{} sh -c '
  set -- {}
  kind=$1; shift
  if [ "$kind" = "pose" ]; then
    .venv/bin/python pose_sensitivity.py "$@" --policy '"$M"' --episodes 10 --out outputs/pose_lora_v2 2>&1 | grep -E "^SUMMARY|Traceback"
  else
    .venv/bin/python switch_experiment.py "$@" --policy '"$M"' --strategy flush --episodes 10 --out outputs/switch_lora_v2 2>&1 | grep -E "^SUMMARY|Traceback"
  fi'
log "DONE"
