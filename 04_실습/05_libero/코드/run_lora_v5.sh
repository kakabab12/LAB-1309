#!/usr/bin/env bash
# LoRA 5차 — **전문가 시범(하이브리드) + DART + 전체 리허설** (DAgger 0회차)
#
# ── 1~4차가 왜 실패했나 (2026-09-28) ──────────────────────────────────
#   네 번 모두 **정책 자신의 성공을 따라하게** 했다 (self-imitation).
#   정책은 교란된 자세에서 거의 성공하지 못한다 → 따라할 좋은 예가 없다.
#   다시 뽑기로 억지로 모은 성공은 운 좋은 노이즈 덕이라, 따라하면 오히려 망가진다
#   (4차: 학습한 T8 이 교란 없는 상태에서도 90% → 10%).
#
# ── 9/28 결정적 시험: 가르칠 것은 두 가지뿐 ─────────────────────────────
#   전문가가 "바로 세워 내려놓기 → B 의 작업 대상 근처로 데려가기" 만 해 주면
#   **원래 정책**이 나머지를 해낸다: 스토브 50→100% (A8, A4 둘 다), 서랍+그릇 0→60%.
#   → 시범 = [전문가: 내려놓기 + 접근] + [정책: 원래 하던 B]. 전부 B 의 지시문으로 저장
#
# ── 5차에 쓰는 기술 ──────────────────────────────────────────────────
#   ① **하이브리드 전문가**   내려놓기·접근은 스크립트, 그 뒤는 정책 (자기 궤적 위라 잘 한다)
#                             정책이 tries 번 다 실패하면 마지막엔 **재생 전문가**가 끝까지
#   ② **DAgger**  (Ross 2011) 정책이 **실제로 가는 상태(전환 순간)** 에서 시범을 모은다
#   ③ **DART**    (Laskey 2017) 전문가 동작에만 잡음 → 틀어진 상태에서의 교정까지 기록
#   ④ **전체 리허설**         10개 태스크 전부, 교란 없이 (4차는 교란 데이터가 리허설을 압도해 무너졌다)
#   ⑤ **용량**                LoRA 랭크 8 → 16
#   ⑥ **공정한 평가**         원래 정책과 **같은 에피소드(20~29)** 에서 나란히, 짝 비교, 망각부터
#   ⑦ **두 줄 병렬**          GPU 메모리 3.4GB/프로세스 → 수집·평가를 두 줄로 (약 40% 단축)
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
EV="--episodes 10 --start-episode 20"
FAIL="8:0 8:3 8:5 8:7 8:9 4:5 4:9 1:7"
OK="8:1 8:4 1:8 4:1"

eval_forget(){   # $1 정책, $2 태그, $3.. 태스크
  local POL=$1 TAG=$2; shift 2
  for t in "$@"; do
    $PY switch_experiment.py --policy "$POL" --task-a "$t" --strategy none $EV \
      --out outputs/v5eval_forget_$TAG 2>&1 | grep -E "$F"
  done
}
eval_switch(){   # $1 정책, $2 태그, $3.. 쌍
  local POL=$1 TAG=$2; shift 2
  for p in "$@"; do
    $PY switch_experiment.py --policy "$POL" --task-a "${p%:*}" --task-b "${p#*:}" \
      --switch-at grasp:3 --strategy flush $EV --out outputs/v5eval_switch_$TAG 2>&1 | grep -E "$F"
  done
}

# ═══ 줄 1: 전문가 시범 → 방향 틀기 시범 ══════════════════════════════════
(
  log "[1/5] 하이브리드 시범 — 쌍 15개 × 에피소드 0~14, DART 0.15, 정책 2번 + 재생 전문가 1번"
  $PY collect_expert.py --mode hybrid --fallback-replay --tries 3 --dart 0.15 --pairs "$PAIRS" \
    --episodes 15 --out data/expert_v5 2>&1 | grep -E "$F"
  log "[2/5] 방향 틀기 시범 — '계속 쥐는' 행동을 잃지 않게"
  $PY collect_expert.py --mode redirect --dart 0.15 --pairs "8:1,8:4,1:8,1:4,4:8,4:1" \
    --episodes 8 --tries 1 --out data/redirect_v5 2>&1 | grep -E "$F"
  log "줄 1 끝"
) &
L1=$!

# ═══ 줄 2: 리허설 → 원래 정책 평가 (5차와 무관하므로 미리) ═════════════════
(
  log "[3/5] 리허설 — 10개 태스크 전부, 교란 없음, 에피소드 0~19"
  $PY collect_expert.py --mode normal --tasks 0 1 2 3 4 5 6 7 8 9 --episodes 20 \
    --out data/rehearsal_v5 2>&1 | grep -E "$F|저장"
  log "[5a/5] ⭐ 망각 기준 — 원래 정책, 10개 태스크"
  eval_forget HuggingFaceVLA/smolvla_libero base 0 1 2 3 4 5 6 7 8 9
  log "[5b/5] 전환 기준 — 원래 정책"
  eval_switch HuggingFaceVLA/smolvla_libero base $FAIL $OK
  log "줄 2 끝"
) &
L2=$!
wait $L1 $L2

log "[4/5] 학습 — LoRA 랭크 16"
$PY train_lora.py --data data/expert_v5 data/redirect_v5 data/rehearsal_v5 --balance --rehearsal-frac 0.5 \
  --steps 4000 --batch-size 4 --grad-accum 2 --lora-r 16 --lora-alpha 32 --lr 5e-5 \
  --out outputs/lora_v5 2>&1 | grep -E '태스크별|지시문|val_loss|합친|Traceback|Error'

# ═══ 5차 평가 — 두 줄 ═════════════════════════════════════════════════
POL=outputs/lora_v5/merged
( log "[5c/5] ⭐ 망각 — 5차";  eval_forget $POL v5 0 1 2 3 4 5 6 7 8 9 ) &
( log "[5d/5] 전환 — 5차";     eval_switch $POL v5 $FAIL $OK ) &
wait
log "끝 — analyze_v5.py 로 원래 정책과 5차를 나란히 비교"
