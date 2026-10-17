#!/usr/bin/env bash
# 손목 맞추기(⑨) 시험 (10/17 11시): als2 + 정렬·놓기·내려가기·다가가기 (als2apr) 에서 '블록 위에서 못 쥠' 이 많은 항목 +
#   T0 (나빠지지 않는지). 손가락-블록 면 어긋남이 못 쥔 장면 중앙 18° (쥔 장면 4°). 비교 기준: als2apr 같은 항목
set -u
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --log-every 500 --abs-pos --objfeat --feat relfeat --vis-cache cache/vis --frame-stride 3 --sufstate"
DATA="data/piper_al_normal_t data/piper_al_wide_normal_t data/piper_al_switch_t data/piper_al_recovery_t data/piper_al_dagger_bo1_t data/piper_al_dagger_al1_t data/piper_al_dagger_als1_t"
TASKS="0 1 2 3 4 5 6 7 8 9"
PAIRS="8:0 8:3 8:5 8:7 8:9 4:5 4:9 1:7 8:1 8:4 1:8 4:1"
board(){ $PY scoreboard.py --blocks --forget outputs/$1_forget --switch outputs/$1_switch --name "$2" --png outputs/media/score_$1.png --md outputs/piper/score_$1.md | tail -1
         $PY scoreboard.py --blocks --loose --forget outputs/$1_forget --switch outputs/$1_switch --name "$2 판 안 기준" --png outputs/media/score_$1_loose.png --md outputs/piper/score_$1_loose.md | tail -1; }
gpu_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
LOCK=outputs/piper/gpu_launch.lock
# 항목 하나 = 평가 한 번. mkdir 로 먼저 잡은 일꾼만 돈다. 띄우기는 자물쇠 + 남은 GPU 메모리 2.8GB, 띄운 뒤 2분 반 기다림
job(){ local q=$1 pol=$2 o=$3 opt=$4 it=$5
  mkdir outputs/piper/claim_$q/$it 2>/dev/null || return 0
  until mkdir $LOCK 2>/dev/null; do sleep 15; done
  while [ "$(gpu_free)" -lt 2800 ]; do sleep 30; done
  if [ "${it#*:}" = "$it" ]; then
    ( $PY piper_sim/eval_piper.py --policy $pol --task-a $it --strategy none --episodes 20 --start-episode 1000 $EVQ $opt --out outputs/${o}_forget 2>&1 | grep -E "$F" ) & local E=$!
  else
    ( $PY piper_sim/eval_piper.py --policy $pol --task-a ${it%:*} --task-b ${it#*:} --switch-at grasp:3 --strategy keep --episodes 20 --start-episode 1000 $EVQ $opt --out outputs/${o}_switch 2>&1 | grep -E "$F" ) & local E=$!
  fi
  sleep 150; rmdir $LOCK
  wait $E; }
queue(){ local q=$1 pol=$2 o=$3 opt=$4; mkdir -p outputs/piper/claim_$q
  for it in $PAIRS $TASKS; do job $q $pol $o "$opt" $it; done; }

rm -rf outputs/piper/claim_yaw $LOCK
TASKS="0 3 9"; PAIRS="8:0 8:7 8:9 8:5"
A2=outputs/piper_als2_model/merged
log "[als2 + 정렬·놓기·내려가기·다가가기·손목 맞추기] 7항목 (일꾼 4개)"
WS=""
for k in 1 2 3 4; do ( queue yaw $A2 piper_als2yaw "--approach-assist --descend-assist --yaw-assist" ) & WS="$WS $!"; sleep 30; done
wait $WS
log "끝"
