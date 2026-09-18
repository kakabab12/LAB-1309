#!/usr/bin/env bash
# LoRA 3차 — **전환 자세 자체**를 많이 모아서 학습한다
#
# 설계 근거 (2026-09-18 저녁)
#   · 전환 자세는 초기 위치에서 27cm 떨어져 있지만, **같은 거리의 임의 교란보다 21~26%p 쉽다**
#     → 전환 자세는 "먼데 **자연스러운**" 자세다 (태스크를 수행하다 도달했으므로)
#   · LoRA 2차는 8~20cm **임의 교란**으로 학습했다 → 성질이 달라 전이가 안 됐을 수 있다
#   · 그러므로 **전환 자세에서 시작하는 성공 궤적**을 많이 모으는 것이 맞다
#   · 전환 성공률이 약 32% 이므로, 40 에피소드당 약 13개가 모인다
#
# 2차와의 차이
#   2차: 정상 궤적 323 + 전환 36 + hindsight 47 + 먼 임의교란 36  (전환 데이터가 적었다)
#   3차: 전환·hindsight 를 **5배 이상**으로. 임의 교란은 보조로만
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
W=${1:-0}
TAG=v3
DATA=data/$TAG
OUT=outputs/lora_$TAG
log(){ echo "=== $(date +%H:%M:%S) $*"; }
F='STATS|Traceback|Error'
if [ "$W" != "0" ]; then
  log "PID $W 대기"
  while kill -0 "$W" 2>/dev/null; do sleep 120; done
fi

log "[1/4] 전환 자세 데이터 수집 (hindsight 모드 — 성공과 '우연히 이룬 목표'를 모두 저장)"
for p in 8:7 4:7 1:5 2:5; do
  for k in grasp:3 grasp:20; do
    $PY collect_robust_data.py --mode hindsight --tasks "${p%:*}" --task-b "${p#*:}" \
      --switch-at "$k" --episodes 40 --start-episode 300 --out "$DATA" 2>&1 \
      | grep -E "$F" | grep -v EGL
  done
done
log "전환 데이터: $(ls $DATA/episodes/S*.npz 2>/dev/null | wc -l) 성공 + $(ls $DATA/episodes/H*.npz 2>/dev/null | wc -l) hindsight"

log "[2/4] 보조: 먼 임의 교란 (20~28cm) — 전환 자세와 성질이 다르지만 범위를 넓히는 효과 확인용"
for t in 1 4 5 7 8; do
  $PY collect_robust_data.py --tasks $t --mode pose --episodes 25 --start-episode 400 \
    --min-offset-cm 20 --max-offset-cm 28 --max-yaw-deg 15 --out "$DATA" 2>&1 \
    | grep -E "$F" | grep -v EGL
done
CNT=$(ls "$DATA"/episodes/*.npz 2>/dev/null | wc -l)
log "수집 합계: $CNT 에피소드"
[ "$CNT" -lt 60 ] && { log "너무 적다 — 중단"; exit 2; }

log "[3/4] LoRA 학습 (기존 + 새 전환 데이터, 균형 샘플링)"
$PY train_lora.py --data data/robust data/v2 "$DATA" --balance --steps 3000 \
  --batch-size 4 --grad-accum 2 --lr 5e-5 --lora-r 8 --lora-alpha 16 --out "$OUT" 2>&1 \
  | grep -E 'LoRA|에피소드|태스크별|val_loss|합친|Traceback|Error'

M=$OUT/merged
log "[4/4] 평가 — 전환 + 망각 확인"
for p in 8:7 4:7 1:5 2:5; do
  for k in grasp:3 grasp:20; do
    $PY switch_experiment.py --policy "$M" --task-a "${p%:*}" --task-b "${p#*:}" --switch-at "$k" \
      --strategy flush --episodes 10 --out "outputs/switch_$TAG" 2>&1 | grep -E 'SUMMARY|Traceback|Error'
  done
done
for tk in 7 1 5; do
  $PY pose_sensitivity.py --policy "$M" --task "$tk" --episodes 10 --out "outputs/pose_$TAG" 2>&1 \
    | grep -E 'SUMMARY|Traceback|Error'
done
log "끝 — analyze_lora_versions.py 에 v3 추가해서 비교할 것"
