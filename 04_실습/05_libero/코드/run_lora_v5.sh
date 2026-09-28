#!/usr/bin/env bash
# LoRA 5차 — **스크립트 전문가 시범 + DART + 전체 리허설** (DAgger 0회차)
#
# ── 1~4차가 왜 실패했나 (2026-09-28) ──────────────────────────────────
#   네 번 모두 **정책 자신의 성공을 따라하게** 했다 (self-imitation).
#   정책은 교란된 자세에서 거의 성공하지 못한다 → 따라할 좋은 예가 없다.
#   다시 뽑기로 억지로 모은 성공은 운 좋은 노이즈 덕이라, 따라하면 오히려 망가진다
#   (4차: 학습한 T8 이 교란 없는 상태에서도 90% → 10%).
#
# ── 5차에 쓰는 기술 ──────────────────────────────────────────────────
#   ① **궤적 재생 전문가**   정책이 원래 성공한 조작을 녹화해 두고, 전환 순간부터
#                             바로 세워 내려놓기 → 그 궤적을 닫힌 고리로 재생.
#                             조작마다 손으로 짤 필요 없이 서랍·스토브·밀기·와인병을 다 다룬다
#   ② **DAgger**  (Ross 2011) 정책이 **실제로 가는 상태(전환 순간)** 에서 시범을 모은다
#   ③ **DART**    (Laskey 2017) 시범 중 실행 동작에만 잡음 → 틀어진 상태에서의 교정까지 기록
#   ④ **전체 리허설**         10개 태스크 전부. 태스크 묶음마다 리허설이 시범만큼 있게
#                             (4차는 T8 묶음 안에서 교란 데이터가 리허설을 압도해 무너졌다)
#   ⑤ **용량**                LoRA 랭크 8 → 16 (새 행동 — 내려놓기·재진입 — 을 배울 여유)
#   ⑥ **공정한 평가**         원래 정책과 **같은 에피소드(20~29)** 에서 나란히, 짝 비교.
#                             망각부터 확인한다 — 여기서 무너지면 나머지는 의미가 없다
#
# 쓰는 법:  ./run_lora_v5.sh [대기할 PID]
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
F='SUMMARY|STATS|Traceback|Error|  =='
log(){ echo "=== $(date +%H:%M:%S) $*"; }
if [ $# -ge 1 ]; then
  log "PID $1 대기"
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi

# 그릇을 든 A → 내려놓고 다른 조작이 필요한 B. 원래 정책이 실패하던 쌍들이다
PAIRS="8:0,8:3,8:5,8:7,8:9,4:0,4:3,4:5,4:7,4:9,1:0,1:3,1:5,1:7,1:9"

log "[1/5] 전문가 시범 — 재생 전문가, DART 0.15, 쌍 15개 × 에피소드 0~14"
$PY collect_expert.py --mode replay --dart 0.15 --pairs "$PAIRS" --episodes 15 --tries 2 \
  --out data/expert_v5 2>&1 | grep -E "$F"

log "[2/5] 방향 틀기 시범 — '계속 쥐는' 행동을 잃지 않게"
$PY collect_expert.py --mode redirect --dart 0.15 --pairs "8:1,8:4,1:8,1:4,4:8,4:1" \
  --episodes 8 --tries 1 --out data/redirect_v5 2>&1 | grep -E "$F"

log "[3/5] 리허설 — 10개 태스크 전부, 교란 없음, 에피소드 0~19"
$PY collect_expert.py --mode normal --tasks 0 1 2 3 4 5 6 7 8 9 --episodes 20 \
  --out data/rehearsal_v5 2>&1 | grep -E "$F|저장"

log "[4/5] 학습 — LoRA 랭크 16"
$PY train_lora.py --data data/expert_v5 data/redirect_v5 data/rehearsal_v5 --balance \
  --steps 4000 --batch-size 4 --grad-accum 2 --lora-r 16 --lora-alpha 32 --lr 5e-5 \
  --out outputs/lora_v5 2>&1 | grep -E '태스크별|val_loss|합친|Traceback|Error'

# ═══ 평가 — 원래 정책과 5차를 **같은 에피소드(20~29)** 에서 나란히 ═══════════════
EV="--episodes 10 --start-episode 20"
for POL in HuggingFaceVLA/smolvla_libero outputs/lora_v5/merged; do
  TAG=$([ "$POL" = "HuggingFaceVLA/smolvla_libero" ] && echo base || echo v5)

  log "[5a/5] ⭐ 망각 — 10개 태스크 교란 없이 ($TAG)"
  for t in 0 1 2 3 4 5 6 7 8 9; do
    $PY switch_experiment.py --policy "$POL" --task-a "$t" --strategy none $EV \
      --out outputs/v5eval_forget_$TAG 2>&1 | grep -E "$F"
  done

  log "[5b/5] 전환 — 원래 실패하던 쌍 ($TAG)"
  for p in 8:0 8:3 8:5 8:7 8:9 4:5 4:9 1:7; do
    $PY switch_experiment.py --policy "$POL" --task-a "${p%:*}" --task-b "${p#*:}" \
      --switch-at grasp:3 --strategy flush $EV --out outputs/v5eval_switch_$TAG 2>&1 | grep -E "$F"
  done

  log "[5c/5] 전환 — 원래 잘 되던 쌍, 나빠지면 안 된다 ($TAG)"
  for p in 8:1 8:4 1:8 4:1; do
    $PY switch_experiment.py --policy "$POL" --task-a "${p%:*}" --task-b "${p#*:}" \
      --switch-at grasp:3 --strategy flush $EV --out outputs/v5eval_switch_$TAG 2>&1 | grep -E "$F"
  done
done
log "끝 — analyze_v5.py 로 원래 정책과 5차를 나란히 비교"
