#!/usr/bin/env bash
# LoRA 3차 — 학습 교란 범위를 전환 시점(27cm)까지 넓힌다
#
# 근거 (2026-09-18)
#   · 쉬운 쌍의 전환 난이도는 "팔이 27cm 떨어진 것"으로 거의 완전히 설명된다
#   · LoRA 2차는 8~20cm 로 학습했다 → 범위가 부족했을 수 있다
#   · 20cm 에서도 50% 성공하므로 그 범위의 데이터는 모을 수 있다
#
# 수집 범위는 경계 실험 결과(outputs/report/boundary.json)를 보고 정한다.
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

RANGE=$($PY - <<'PY'
import json, os
lo, hi = 20, 30
try:
    c = json.load(open("outputs/report/boundary.json"))["off_curve"]
    ok = sorted(float(k) for k, v in c.items() if v >= 0.2)
    if ok:
        hi = max(30.0, max(ok) + 5)      # 성공하는 최대 교란보다 5cm 더
        lo = max(12.0, max(ok) - 8)      # 그 아래 8cm 부터
except Exception:
    pass
print(f"{lo:g} {hi:g}")
PY
)
LO=$(echo "$RANGE" | awk '{print $1}'); HI=$(echo "$RANGE" | awk '{print $2}')
log "수집 범위: ${LO}~${HI}cm (경계 실험 결과 기반)"

log "[1/3] 먼 교란 데이터 수집 (성공한 것만 저장)"
for t in 0 1 2 4 5 7 8; do
  $PY collect_robust_data.py --tasks $t --mode pose --episodes 40 --start-episode 300 \
    --min-offset-cm "$LO" --max-offset-cm "$HI" --max-yaw-deg 15 --out "$DATA" 2>&1 \
    | grep -E "$F" | grep -v EGL
done
CNT=$(ls "$DATA"/episodes/*.npz 2>/dev/null | wc -l)
log "수집된 에피소드: $CNT"
[ "$CNT" -lt 30 ] && { log "너무 적다 — 중단"; exit 2; }

log "[2/3] LoRA 학습 (기존 데이터 + 먼 교란, 균형 샘플링)"
$PY train_lora.py --data data/robust data/v2 "$DATA" --balance --steps 2500 \
  --batch-size 4 --grad-accum 2 --lr 5e-5 --lora-r 8 --lora-alpha 16 --out "$OUT" 2>&1 \
  | grep -E 'LoRA|에피소드|태스크별|val_loss|합친|Traceback|Error'

M=$OUT/merged
log "[3/3] 평가 — 전환 + 망각 확인"
for p in 8:7 4:7 1:5 2:5; do
  for k in grasp:3 grasp:20; do
    $PY switch_experiment.py --policy "$M" --task-a "${p%:*}" --task-b "${p#*:}" --switch-at "$k" \
      --strategy flush --episodes 10 --out "outputs/switch_$TAG" 2>&1 | grep -E 'SUMMARY|Traceback|Error'
  done
done
for tk in 7 1 5; do
  $PY pose_sensitivity.py --policy "$M" --task "$tk" --episodes 10 --out "outputs/pose_$TAG" 2>&1 \
    | grep -E 'SUMMARY|Traceback|Error'
  $PY pose_sensitivity.py --policy "$M" --task "$tk" --offset-cm 20 --episodes 10 \
    --out "outputs/pose_$TAG" 2>&1 | grep -E 'SUMMARY|Traceback|Error'
done
log "끝 — analyze_lora_versions.py 에 v3 추가해서 비교할 것"
