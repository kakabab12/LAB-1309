#!/usr/bin/env bash
# 최종 평가 — 학습에 안 쓴 새 배치에서 항목마다 N 장면 (2026-10-02, 사용자: "시간 오래 걸려도 되니까 장면 최대한 많이")
#   ./run_final_eval.sh POLICY OUTPREFIX N [A2C2DIR] [DIMS] [NA]
#   예: ./run_final_eval.sh outputs/v6e_model/merged final_v6e 100 outputs/a2c2_v6e all 1
# 장면 1000 ~ 1000+N-1. 95% 달성 = N 장면 중 0.95N 이상. GPU 프로세스 3개.
cd "$(dirname "$0")"
POL=$1; OUT=$2; N=${3:-100}; HEAD=${4:-}; DIMS=${5:-all}; NA=${6:-1}
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
EV="--episodes $N --start-episode 1000 --latency-steps 11 --ttrtc --n-action-steps $NA"
[ -n "$HEAD" ] && EV="$EV --a2c2 $HEAD --a2c2-dims $DIMS"
run_t(){ for t in "$@"; do $PY switch_experiment.py --policy $POL --task-a $t --strategy none $EV --out outputs/${OUT}_forget 2>&1 | grep -E "$F"; done; }
run_p(){ for p in "$@"; do $PY switch_experiment.py --policy $POL --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy keep $EV --out outputs/${OUT}_switch 2>&1 | grep -E "$F"; done; }
echo "=== $(date +%m/%d\ %H:%M) 최종 평가 시작: $POL, 장면 $N 개, 보정 ${HEAD:-없음}"
( run_t 0 1 2 3 4 5 6 7 8 9 ) &
( run_p 8:0 8:3 8:5 8:7 8:9 4:5 ) &
( run_p 4:9 1:7 8:1 8:4 1:8 4:1 ) &
wait
$PY scoreboard.py --forget outputs/${OUT}_forget --switch outputs/${OUT}_switch --name "$OUT (새 배치 $N 장면, 지연 11)" \
  --png outputs/media/score_${OUT}.png --md outputs/v6/score_${OUT}.md | tail -1
echo "=== $(date +%m/%d\ %H:%M) 최종 평가 끝"
