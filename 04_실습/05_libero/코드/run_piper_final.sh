#!/usr/bin/env bash
# PiPER 최종 평가 (10/3 준비): 학습에 안 쓴 새 배치 1000~(1000+N-1), 항목마다 N 장면, 실제 배치 조건
#   ./run_piper_final.sh outputs/piper_s3_model/merged piper_final 100 [A2C2DIR]
#   95% 달성 = N 장면 중 0.95N 이상. GPU 프로세스 4개 (학습이 없을 때 돌릴 것)
cd "$(dirname "$0")"
POL=$1; OUT=$2; N=${3:-100}; HEAD=${4:-}
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
EV="--episodes $N --start-episode 1000 --latency-steps 11 --ttrtc --n-action-steps 1"
[ -n "$HEAD" ] && EV="$EV --a2c2 $HEAD"
run_t(){ for t in "$@"; do $PY piper_sim/eval_piper.py --policy $POL --task-a $t --strategy none $EV --out outputs/${OUT}_forget 2>&1 | grep -E "$F"; done; }
run_p(){ for p in "$@"; do $PY piper_sim/eval_piper.py --policy $POL --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy keep $EV --out outputs/${OUT}_switch 2>&1 | grep -E "$F"; done; }
echo "=== $(date +%m/%d\ %H:%M) PiPER 최종 평가: $POL, $N 장면, 보정 ${HEAD:-없음}"
( run_t 0 1 2 3 4 ) & ( run_t 5 6 7 8 9 ) & ( run_p 8:0 8:3 8:5 8:7 8:9 4:5 ) & ( run_p 4:9 1:7 8:1 8:4 1:8 4:1 ) &
wait
$PY scoreboard.py --forget outputs/${OUT}_forget --switch outputs/${OUT}_switch --name "$OUT (PiPER 새 배치 $N 장면, 지연 0.56초)" \
  --png outputs/media/score_${OUT}.png --md outputs/piper/score_${OUT}.md | tail -1
echo "=== $(date +%m/%d\ %H:%M) 끝"
