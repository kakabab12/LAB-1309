#!/usr/bin/env bash
# 내려놓기 보조 28항목 (10/22 02시): np2 + 막힘 풀기·손목 맞추기(86.4%) 의 일 바꾸기 140장면 중 59장면은 VLA 가 A 를 B 의 놓을 곳에
#   내려놓았고, 그때 B 성공 32/59 (다른 곳 78/81), 돌아오기 17/47 (다른 곳 67/73). 시범은 A 를 쥔 자리에 그대로 내려놓는다.
#   → grasp_assist ⑫ 내려놓기 (--putdown-assist): 쥔 블록이 지시문의 블록이 아니면 빈 자리(처음 자리 우선)로 옮겨 내려놓음.
#   ① run_np2c_round.sh (2275043, np2 단독 평가) 끝 기다림 → ② 일꾼 4개: np2 + 보조 + 막힘 풀기·손목 맞추기 + 내려놓기 28항목 (piper_np2pd)
set -u
cd "$(dirname "$0")"
export PIPER_NOPED=1 PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
TASKS="0 1 2 3 4 5 6 7 8 9"
PAIRS="8:0 8:3 8:5 8:7 8:9 4:5 4:9 1:7 8:1 8:4 1:8 4:1"
PD="--approach-assist --descend-assist --unjam-assist --yaw-assist --putdown-assist"
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
    ( $PY piper_sim/eval_piper.py --policy $pol --task-a ${it%:*} --task-b ${it#*:} --switch-at grasp:3 --strategy keep --episodes 20 --start-episode 1000 $EVQ $opt --out outputs/${o}_switch 2>&1 | grep -E "$F" ) & local E=$!
  fi
  sleep 150; rmdir $LOCK
  wait $E; }
queue(){ local q=$1 pol=$2 o=$3 opt=$4; mkdir -p outputs/piper/claim_$q
  for it in $PAIRS $TASKS; do job $q $pol $o "$opt" $it; done; }

while kill -0 2275043 2>/dev/null; do sleep 60; done
C=outputs/piper_np2_model/merged
log "[np2 + 보조 + 막힘 풀기·손목 맞추기 + 내려놓기 28항목] (일꾼 4개)"
rm -rf outputs/piper/claim_n2d $LOCK
WS=""
for k in 1 2 3 4; do ( queue n2d $C piper_np2pd "$PD" ) & WS="$WS $!"; sleep 30; done
wait $WS
board piper_np2pd "받침대 없음 np2 + 보조 + 막힘 풀기·손목 맞추기 + 내려놓기 (지연 9)"
$PY fail_classify.py outputs/piper_np2pd --md outputs/piper/fail_piper_np2pd.md > /dev/null
log "끝"
