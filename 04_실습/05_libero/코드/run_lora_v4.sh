#!/usr/bin/env bash
# LoRA 4차 — **"교란된 자세에서 물체 집기"** 하나만 학습한다
#
# 왜 이번엔 다른가 (2026-09-23 진단)
#   전환 뒤 팔은 목표 물체 **5cm 까지 다가간다** (10cm 안 100%). 그런데 못 집는다.
#   **이동은 살아 있고 집기만 무너졌다.**
#     B 가 가구·기구(집을 필요 없음) → 44.6%
#     B 가 물체(집어야 함)          →  1.4%
#   9/16 자세 민감도와도 맞물린다: 손목 30도에서 물체 집는 태스크가 0%.
#
#   1~3차는 전부 **"전환" 전체**를 학습시키려 했고 32→30% 로 안 움직였다.
#   4차는 **하나의 기술(교란된 자세에서 집기)** 만 겨냥한다.
#
# ⭐ 설계에서 가장 중요한 차이 — **중간 지표를 먼저 본다**
#   1~3차는 곧바로 "전환 성공률"로 평가했다. 그래서 왜 안 되는지 알 수 없었다.
#   4차는 **부분 기술이 늘었는지 먼저 확인**한다:
#       ① 교란 27cm 에서 물체 집는 태스크의 성공률이 올랐는가   ← 이게 안 오르면 학습 실패
#       ② 그 다음에야 전환 성공률을 본다                       ← 이게 안 오르면 진단이 틀린 것
#   둘을 나눠야 "학습이 안 된 것"과 "진단이 틀린 것"을 구분할 수 있다.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
F='SUMMARY|STATS|Traceback|Error'
log(){ echo "=== $(date +%H:%M:%S) $*"; }

if [ $# -ge 1 ]; then
  log "PID $1 대기"
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi

# ── 1. 데이터 수집 — 집는 태스크만, 전환과 같은 교란 범위 ───────────────
# 태스크 1·8·9 = 물체를 집어야 하는 것. 0·7 은 제외 (이미 잘 된다)
# 교란 15~30cm: 전환 시점(27cm)을 **포함**한다.
#   ⚠️ 2차는 8~20cm 로 학습하고 8cm 까지만 검증해서 27cm 로 전이되지 않았다.
# --tries 8: 성공률이 낮은 구간이라 다시 뽑지 않으면 데이터가 안 모인다
log "[1/4] 교란된 자세에서 '집기' 데이터 수집 (태스크 1·8·9, 15~30cm)"
$PY collect_robust_data.py --mode pose --tasks 1 8 9 \
  --min-offset-cm 15 --max-offset-cm 30 --max-yaw-deg 10 \
  --episodes 40 --tries 8 --out data/grasp_far 2>&1 | grep -E "$F"

# 망각 방지용 리허설: 교란 없는 정상 데이터도 섞는다
log "[2/4] 리허설 데이터 (교란 없음) — 망각 방지"
$PY collect_robust_data.py --mode pose --tasks 0 1 5 7 8 9 \
  --min-offset-cm 0 --max-offset-cm 4 --max-yaw-deg 5 \
  --episodes 12 --tries 2 --out data/rehearsal_v4 2>&1 | grep -E "$F"

# ── 2. 학습 ──────────────────────────────────────────────────────────
# rank 8 / lr 5e-5: 2차에서 망각이 줄었던 설정 (LoRA-망각 논문)
log "[3/4] LoRA 학습"
$PY train_lora.py --data data/grasp_far data/rehearsal_v4 --balance \
  --steps 3000 --batch-size 4 --grad-accum 2 --lora-r 8 --lora-alpha 16 --lr 5e-5 --out outputs/lora_v4 2>&1 | tail -25

# ── 3. ⭐ 중간 지표 먼저 — 부분 기술이 늘었는가 ────────────────────────
log "[4/4a] ⭐ 부분 기술 검증 — 교란된 자세에서 집기가 늘었는가"
for t in 1 8 9; do
  for o in 0 12 20 27; do
    $PY pose_sensitivity.py --task "$t" --offset-cm "$o" --episodes 10 \
      --policy outputs/lora_v4/merged --out outputs/pose_v4 2>&1 | grep -E "$F"
  done
done
log "[4/4b] 망각 확인 — 안 집는 태스크가 나빠지지 않았는가"
for t in 0 7; do
  $PY pose_sensitivity.py --task "$t" --offset-cm 0 --episodes 10 \
    --policy outputs/lora_v4/merged --out outputs/pose_v4 2>&1 | grep -E "$F"
done

log "[4/4c] 그 다음에야 전환 — B 가 물체인 쌍과 아닌 쌍을 나눠서"
for p in 8:3 4:9 2:9; do   # B 가 물체 (현재 0%)
  $PY switch_experiment.py --task-a "${p%:*}" --task-b "${p#*:}" --switch-at grasp:3 \
    --strategy flush --policy outputs/lora_v4/merged --episodes 10 --out outputs/switch_v4 2>&1 | grep -E "$F"
done
for p in 8:0 8:7; do       # B 가 가구·기구 (현재 60/50%) — 나빠지면 안 된다
  $PY switch_experiment.py --task-a "${p%:*}" --task-b "${p#*:}" --switch-at grasp:3 \
    --strategy flush --policy outputs/lora_v4/merged --episodes 10 --out outputs/switch_v4 2>&1 | grep -E "$F"
done

log "끝 — 읽는 순서: [4/4a] 부분 기술 → [4/4b] 망각 → [4/4c] 전환"
