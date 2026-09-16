#!/usr/bin/env bash
# 전환 실험 스윕: 태스크 쌍 4 × 전환 시점 3 × 전략 5 × 에피소드 10  (3개 프로세스 병렬)
set -u
cd "$(dirname "$0")"
PAIRS="8:7 4:7 1:5 2:5"      # A:B  (A=물체를 잡는 태스크, B=충돌 없는 태스크)
TIMINGS="step:15 grasp:3 grasp:20"   # 접근 중 / 잡은 직후 / 들고 이동 중
JOBS=outputs/switch_jobs.txt
: > $JOBS
for s in none flush keep blend retreat; do   # none 먼저 (반응 시간 계산 기준)
  for p in $PAIRS; do for k in $TIMINGS; do
    echo "--task-a ${p%:*} --task-b ${p#*:} --switch-at $k --strategy $s" >> $JOBS
  done; done
done
cat $JOBS | xargs -P 3 -I{} sh -c '.venv/bin/python switch_experiment.py {} --episodes 10 --video --out outputs/switch 2>&1 | grep -E "^SUMMARY|Traceback|Error"'
echo DONE
