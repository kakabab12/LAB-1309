#!/usr/bin/env bash
# run_np2_round.sh 의 평가 부분을 바꿈 (10/21 18:40): 학습(2174024)은 그대로 두고 본체만 끄고 이것이 이어받는다.
#   np1 + 보조 '블록 위에서 못 쥠' 장면을 다시 돌려 보니 손가락이 블록 윗면·모서리에 걸린 뒤 내려가기 재촉이 계속 눌러 손목 관절이
#   한계에 박혔다 → 막힘 풀기(⑪) + 6번 관절 한계를 보는 손목 맞추기(⑨). 실패 장면 그대로 다시 돌리기에서 고쳐진 쪽이 많아
#   (run_unjam_test.sh) 28항목 전체로 잰다. 원래 성공하던 장면이 망가지는지는 이것으로만 알 수 있다.
#   ① np2 학습 끝 기다림 → ② 일꾼 4개: np2 + 보조 28항목 → np2 + 보조 + 막힘 풀기 + 손목 맞추기 28항목 → np2 단독 28항목
set -u
cd "$(dirname "$0")"
export PIPER_NOPED=1 PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
TASKS="0 1 2 3 4 5 6 7 8 9"
PAIRS="8:0 8:3 8:5 8:7 8:9 4:5 4:9 1:7 8:1 8:4 1:8 4:1"
AS="--approach-assist --descend-assist"
UJ="$AS --unjam-assist --yaw-assist"
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

# ① np2 학습 끝 (PID 로 기다림)
while kill -0 2174024 2>/dev/null; do sleep 60; done
C=outputs/piper_np2_model/merged
[ -f $C/sufstate ] && [ -f $C/relfeat ] || { log "np2 모델 없음 또는 relfeat 표시 없음 — 멈춤"; exit 1; }
log "np2 학습 끝"

# ② 일꾼 4개
log "[np2 + 보조 28항목] → [np2 + 보조 + 막힘 풀기 + 손목 맞추기 28항목] → [np2 단독 28항목] (일꾼 4개)"
rm -rf outputs/piper/claim_n2p outputs/piper/claim_n2u outputs/piper/claim_n2h $LOCK
worker(){ queue n2p $C piper_np2apr "$AS"; queue n2u $C piper_np2uj "$UJ"; queue n2h $C piper_np2h ""; }
WS=""
for k in 1 2 3 4; do ( worker ) & WS="$WS $!"; sleep 30; done
wait $WS
board piper_np2apr "받침대 없음 np2 + 정렬·놓기·내려가기·다가가기 (지연 9)"
board piper_np2uj "받침대 없음 np2 + 보조 + 막힘 풀기·손목 맞추기 (지연 9)"
board piper_np2h "받침대 없음 np2 단독 (지연 9)"
for o in piper_np2apr piper_np2uj piper_np2h; do PIPER_BLOCKS=1 $PY fail_classify.py outputs/$o --md outputs/piper/fail_$o.md > /dev/null; done
log "끝"
