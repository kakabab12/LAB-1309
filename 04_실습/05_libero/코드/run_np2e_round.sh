#!/usr/bin/env bash
# 넓은 변환식 28항목 (10/22 08시): 보조가 쓰는 정면 카메라 변환식을 책상 전체로 맞춘 것으로 (calib_front_wide.py).
#   예전 식(블록마다 처음 배치로만 맞춤)은 블록이 제자리를 떠나면 평균 1.8cm·최대 13.6cm 틀렸다 (새 식 평균 0.28cm).
#   np2 + 보조 5가지 (93.8%, piper_np2pd) 와 변환식만 다름 → piper_np2pw. 학생 모델 입력(relfeat)은 예전 식 그대로 (학습과 같게)
set -u
cd "$(dirname "$0")"
export PIPER_NOPED=1 PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl PIPER_FRONT_CALIB=piper_sim/front_calib_wide.json
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


C=outputs/piper_np2_model/merged
log "[np2 + 보조 5가지, 넓게 맞춘 변환식 28항목] (일꾼 4개)"
rm -rf outputs/piper/claim_n2w $LOCK
WS=""
for k in 1 2 3 4; do ( queue n2w $C piper_np2pw "$PD" ) & WS="$WS $!"; sleep 30; done
wait $WS
board piper_np2pw "받침대 없음 np2 + 보조 5가지, 넓은 변환식 (지연 9)"
$PY fail_classify.py outputs/piper_np2pw --md outputs/piper/fail_piper_np2pw.md > /dev/null
log "끝"
