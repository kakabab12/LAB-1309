#!/usr/bin/env bash
# LoRA 4차 — **"교란된 자세에서 물체 집기"** 를 단계적으로 넓혀 가며 학습
#
# ── 왜 이번엔 다른가 (2026-09-23 진단) ─────────────────────────────────
#   전환 뒤 팔은 목표 물체 **5cm 까지 다가간다** (10cm 안 100%). 다시 집으려 시도도 한다(8/10).
#   그런데 못 집는다. **이동은 살아 있고 정밀한 집기만 무너졌다.**
#     B 가 가구·기구(집을 필요 없음) → 44.6%
#     B 가 물체(집어야 함)          →  1.4%
#
#   교란별로 재 보면 선명하다 (태스크 1 = 물체를 집어야 함):
#       0cm 100%  →  12cm 30%  →  20cm 10%  →  **27cm 0%**
#   전환 시점의 이탈이 바로 **27cm** 다.
#
#   1~3차는 전부 "전환" 전체를 학습시키려 했고 32→30% 로 안 움직였다.
#   4차는 **하나의 기술(교란된 자세에서 집기)** 만 겨냥한다.
#
# ── ⚠️ 그냥 27cm 데이터를 모으려 하면 실패한다 ─────────────────────────
#   27cm 에서 성공률이 0% 다. **다시 뽑아도(--tries) 데이터가 하나도 안 모인다.**
#   그러면 15~20cm 데이터만 쌓여 **LoRA 2차와 똑같은 함정**에 빠진다
#   (2차: 8~20cm 로 학습 → 8cm 까지만 검증 → 27cm 로 전이 안 됨).
#
#   그래서 **단계적 확장(역커리큘럼)** 으로 간다:
#     1단계  12~20cm 로 학습   ← 여기선 성공률 10~30% 라 데이터가 모인다
#     2단계  좋아진 정책으로 20~28cm 를 다시 모아 학습   ← 1단계가 성공해야 가능
#   각 단계마다 **집기가 실제로 늘었는지 확인**하고 넘어간다.
#   📚 [Reverse Curriculum](../../../02_논문노트/Reverse-Curriculum.md)
#
# ── ⭐ 설계에서 가장 중요한 차이 — 중간 지표를 먼저 본다 ───────────────
#   1~3차는 곧바로 "전환 성공률"로 평가해서, 안 될 때 **왜 안 되는지 알 수 없었다.**
#     ① 교란된 자세에서 집기가 늘었는가   ← 안 오르면 **학습이 안 된 것**
#     ② 그 다음에야 전환 성공률           ← 안 오르면 **진단이 틀린 것**
#   둘을 나눠야 다음에 무엇을 고칠지가 정해진다.
#   그리고 ① 만 올라도 그 자체로 결과다.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
F='SUMMARY|STATS|Traceback|Error'
log(){ echo "=== $(date +%H:%M:%S) $*"; }

if [ $# -ge 1 ]; then
  log "PID $1 대기"
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi

# 집는 태스크만 쓴다 (0·7 은 이미 잘 되므로 제외). 1·8·9 = 물체를 집어야 함.
GRASP_TASKS="1 8 9"

# 부분 기술을 재는 함수 — 매 단계마다 같은 조건으로 잰다
measure() {   # $1=정책경로  $2=출력폴더
  for t in $GRASP_TASKS; do
    for o in 0 12 20 27; do
      $PY pose_sensitivity.py --task "$t" --offset-cm "$o" --episodes 10 \
        --policy "$1" --out "$2" 2>&1 | grep -E "$F"
    done
  done
  for t in 0 7; do   # 망각 확인 (집을 필요 없는 태스크)
    $PY pose_sensitivity.py --task "$t" --offset-cm 0 --episodes 10 \
      --policy "$1" --out "$2" 2>&1 | grep -E "$F"
  done
}

# ══ 1단계: 12~20cm ═══════════════════════════════════════════════════
log "[1-1] 데이터 수집 12~20cm (성공률 10~30% 구간, --tries 10)"
$PY collect_robust_data.py --mode pose --tasks $GRASP_TASKS \
  --min-offset-cm 12 --max-offset-cm 20 --max-yaw-deg 10 \
  --episodes 40 --tries 10 --out data/grasp_s1 2>&1 | grep -E "$F"

log "[1-2] 리허설 데이터 (교란 없음) — 망각 방지"
$PY collect_robust_data.py --mode pose --tasks 0 1 5 7 8 9 \
  --min-offset-cm 0 --max-offset-cm 4 --max-yaw-deg 5 \
  --episodes 12 --tries 2 --out data/rehearsal_v4 2>&1 | grep -E "$F"

log "[1-3] 학습 (1단계)"
$PY train_lora.py --data data/grasp_s1 data/rehearsal_v4 --balance \
  --steps 3000 --batch-size 4 --grad-accum 2 --lora-r 8 --lora-alpha 16 --lr 5e-5 \
  --out outputs/lora_v4s1 2>&1 | tail -20

log "[1-4] ⭐ 부분 기술 확인 (1단계) — 집기가 늘었는가"
measure outputs/lora_v4s1/merged outputs/pose_v4s1

# ══ 2단계: 20~28cm (1단계 정책으로 데이터를 모은다) ═══════════════════
log "[2-1] 데이터 수집 20~28cm — **1단계 정책으로** 모은다"
echo "    (원본으로는 이 구간에서 0% 라 데이터가 안 모인다. 1단계가 성공해야 가능하다)"
$PY collect_robust_data.py --mode pose --tasks $GRASP_TASKS \
  --policy outputs/lora_v4s1/merged \
  --min-offset-cm 20 --max-offset-cm 28 --max-yaw-deg 10 \
  --episodes 40 --tries 15 --out data/grasp_s2 2>&1 | grep -E "$F"

log "[2-2] 학습 (2단계) — 1·2단계 데이터 + 리허설"
$PY train_lora.py --data data/grasp_s1 data/grasp_s2 data/rehearsal_v4 --balance \
  --steps 3000 --batch-size 4 --grad-accum 2 --lora-r 8 --lora-alpha 16 --lr 5e-5 \
  --out outputs/lora_v4 2>&1 | tail -20

log "[2-3] ⭐ 부분 기술 확인 (2단계)"
measure outputs/lora_v4/merged outputs/pose_v4

# ══ 3단계: 그 다음에야 전환 ══════════════════════════════════════════
log "[3-1] 전환 — B 가 **물체**인 쌍 (현재 0%)"
for p in 8:3 4:9 2:9 8:5; do
  $PY switch_experiment.py --task-a "${p%:*}" --task-b "${p#*:}" --switch-at grasp:3 \
    --strategy flush --policy outputs/lora_v4/merged --episodes 10 \
    --out outputs/switch_v4 2>&1 | grep -E "$F"
done

log "[3-2] 전환 — B 가 **가구·기구**인 쌍 (현재 60/50%) — 나빠지면 안 된다"
for p in 8:0 8:7 4:0; do
  $PY switch_experiment.py --task-a "${p%:*}" --task-b "${p#*:}" --switch-at grasp:3 \
    --strategy flush --policy outputs/lora_v4/merged --episodes 10 \
    --out outputs/switch_v4 2>&1 | grep -E "$F"
done

log "끝 — 읽는 순서: [1-4] → [2-3] 부분 기술이 올랐나 → [3-1]/[3-2] 전환"
