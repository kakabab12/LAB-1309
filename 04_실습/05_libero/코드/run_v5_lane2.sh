#!/usr/bin/env bash
# LoRA 5차 — 줄 2: 리허설 수집 → 원래 정책 평가 (5차와 무관하므로 먼저 돌린다)
# 끝나면 outputs/v5_lane2.done 을 만든다 → run_lora_v5.sh 가 이걸 기다렸다가 학습
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
F='SUMMARY|STATS|Traceback|Error|  =='
log(){ echo "=== $(date +%H:%M:%S) $*"; }
EV="--episodes 10 --start-episode 20"
rm -f outputs/v5_lane2.done

log "[3/5] 리허설 — 10개 태스크 전부, 교란 없음, 에피소드 0~19"
$PY collect_expert.py --mode normal --tasks 0 1 2 3 4 5 6 7 8 9 --episodes 20 \
  --out data/rehearsal_v5 2>&1 | grep -E "$F|저장"
touch outputs/v5_rehearsal.done

log "[5a/5] ⭐ 망각 기준 — 원래 정책, 10개 태스크 (에피소드 20~29)"
for t in 0 1 2 3 4 5 6 7 8 9; do
  $PY switch_experiment.py --policy HuggingFaceVLA/smolvla_libero --task-a "$t" --strategy none $EV \
    --out outputs/v5eval_forget_base 2>&1 | grep -E "$F"
done
log "[5b/5] 전환 기준 — 원래 정책"
for p in 8:0 8:3 8:5 8:7 8:9 4:5 4:9 1:7 8:1 8:4 1:8 4:1; do
  $PY switch_experiment.py --policy HuggingFaceVLA/smolvla_libero --task-a "${p%:*}" --task-b "${p#*:}" \
    --switch-at grasp:3 --strategy flush $EV --out outputs/v5eval_switch_base 2>&1 | grep -E "$F"
done
touch outputs/v5_lane2.done
log "줄 2 끝"
