#!/usr/bin/env bash
# 전체 파이프라인: (ablation 종료 대기) → 데이터 수집 → LoRA 학습 → 평가
# 목표: 전환할 때 초기 자세로 되돌리지 않고도(리셋 금지) 자연스럽게 새 태스크를 수행하게 만들기
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
log(){ echo "=== $(date +%H:%M:%S) $*"; }

log "1/5 ablation 종료 대기"
until grep -q '^DONE' outputs/ablation_sweep.log 2>/dev/null; do sleep 60; done

log "2/5 자세 교란 데이터 수집 (성공한 궤적만)"
J=outputs/collect_pose_jobs.txt; : > $J
for t in 0 1 2 4 5 7 8; do echo "--tasks $t"; done > $J
cat $J | xargs -P 3 -I{} sh -c "$PY collect_robust_data.py {} --mode pose --episodes 40 --max-offset-cm 8 --max-yaw-deg 20 --out data/robust 2>&1 | grep -E 'STATS|Traceback|Error'"

log "3/5 전환 데이터 수집 (리셋 없이 전환해서 성공한 궤적만)"
J=outputs/collect_switch_jobs.txt; : > $J
for p in 8:7 4:7 1:5 2:5; do
  for k in grasp:3 grasp:20; do echo "--tasks ${p%:*} --task-b ${p#*:} --switch-at $k"; done
done > $J
cat $J | xargs -P 3 -I{} sh -c "$PY collect_robust_data.py {} --mode switch --episodes 25 --out data/robust 2>&1 | grep -E 'STATS|Traceback|Error'"
ls data/robust/episodes | wc -l

log "4/5 LoRA 학습"
$PY train_lora.py --data data/robust --steps 2000 --batch-size 4 --grad-accum 2 --lr 1e-4 \
  --out outputs/lora_v1 2>&1 | grep -E '^\{|LoRA|에피소드|저장|Traceback|Error'

M=outputs/lora_v1/merged
log "5/5 평가 (학습된 모델)"
J=outputs/eval_jobs.txt; : > $J
for t in 7 1 5; do
  for o in 0 4 8; do echo "pose --task $t --offset-cm $o"; done
  for y in 10 20 30; do echo "pose --task $t --yaw-deg $y"; done
done > $J
for p in 8:7 4:7 1:5 2:5; do
  for k in grasp:3 grasp:20; do echo "switch --task-a ${p%:*} --task-b ${p#*:} --switch-at $k"; done
done >> $J
cat $J | xargs -P 3 -I{} sh -c '
  set -- {}
  kind=$1; shift
  if [ "$kind" = "pose" ]; then
    .venv/bin/python pose_sensitivity.py "$@" --policy '"$M"' --episodes 10 --out outputs/pose_lora 2>&1 | grep -E "^SUMMARY|Traceback"
  else
    .venv/bin/python switch_experiment.py "$@" --policy '"$M"' --strategy flush --episodes 10 --out outputs/switch_lora 2>&1 | grep -E "^SUMMARY|Traceback"
  fi'
log "DONE"
