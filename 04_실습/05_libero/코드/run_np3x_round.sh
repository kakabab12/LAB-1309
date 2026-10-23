#!/usr/bin/env bash
# np3 + 고친 보조 + ⑲ + ⑨' + ⑰' + ⑳ (10/23 15:15): 사용자 "셋 다 성공도 90 넘기자".
#   A8→B5·A8→B7 40장면 짝 비교 (run_np3x_test.sh): B 실패 8 → 3, 셋 다 32 → 35 (나빠진 장면 0). 나머지 돌아오기 4쌍 → 다른 6쌍 → 혼자 10
#   8:5·8:7 은 run_np3x_test.sh 결과 (outputs/piper_np3x_switch). 일꾼 4개. run_np3s_round.sh(⑲ 만)의 남은 줄은 자리표로 멈춤
set -u
cd "$(dirname "$0")"
export PIPER_NOPED=1 PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl PIPER_FRONT_CALIB=piper_sim/front_calib_wide.json
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph --release-wait 40"
SW="grasp:30"
TASKS="0 1 2 3 4 5 6 7 8 9"
PAIRS="8:9 4:5 4:9 1:7 8:0 8:3 8:1 8:4 1:8 4:1"
FIX="--approach-assist --descend-assist --unjam-assist --yaw-assist --putdown-assist --fresh-held --putdown-near --wrong-guard --mem-refresh --putdown-far --stuck-grasp 40 --yaw-none-reset 10 --mem-tol 0.012 --open-empty"
board(){ $PY scoreboard.py --blocks --forget outputs/$1_forget --switch outputs/$1_switch --name "$2" --png outputs/media/score_$1.png --md outputs/piper/score_$1.md | tail -1
         $PY scoreboard.py --blocks --loose --forget outputs/$1_forget --switch outputs/$1_switch --name "$2 판 안 기준" --png outputs/media/score_$1_loose.png --md outputs/piper/score_$1_loose.md | tail -1; }
gpu_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
LOCK=outputs/piper/gpu_launch.lock
job(){ local q=$1 pol=$2 o=$3 opt=$4 it=$5
  mkdir outputs/piper/claim_$q/$it 2>/dev/null || return 0
  until mkdir $LOCK 2>/dev/null; do sleep 15; done
  while [ "$(gpu_free)" -lt 2800 ]; do sleep 30; done
  if [ "${it#*:}" = "$it" ]; then
    ( $PY piper_sim/eval_piper.py --policy $pol --task-a $it --strategy none --episodes 20 --start-episode 1000 $EVQ $opt --out outputs/${o}_forget 2>&1 | grep -E "$F" ) & local E=$!
  else
    ( $PY piper_sim/eval_piper.py --policy $pol --task-a ${it%:*} --task-b ${it#*:} --switch-at $SW --strategy keep --episodes 20 --start-episode 1000 $EVQ $opt --out outputs/${o}_switch 2>&1 | grep -E "$F" ) & local E=$!
  fi
  sleep 150; rmdir $LOCK
  wait $E; }
queue(){ local q=$1 pol=$2 o=$3 opt=$4 items=$5; mkdir -p outputs/piper/claim_$q
  for it in $items; do job $q $pol $o "$opt" $it; done; }
C2=outputs/piper_np2_model/merged
C3=outputs/piper_np3_model/merged
rm -rf outputs/piper/claim_x3p
log "[새 기준 np3 + 고친 보조 + ⑲ + 새 옵션] 남은 돌아오기 4쌍 먼저 (일꾼 4개)"
worker(){ queue x3p $C3 piper_np3x "$FIX" "$PAIRS $TASKS"; }
WS=""
for k in 1 2 3 4; do ( worker ) & WS="$WS $!"; sleep 30; done
wait $WS
board piper_np3x "새 기준 np3 + 고친 보조 + ⑲⑨'⑰'⑳ (지연 9)"
$PY fail_classify.py outputs/piper_np3x --md outputs/piper/fail_piper_np3x.md > /dev/null
log "끝"
