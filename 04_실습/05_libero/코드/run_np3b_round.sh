#!/usr/bin/env bash
# 새 기준 평가 (10/22 15시): 사용자 "어느 정도 수행했다가 다른 곳에 놨다가 다른 태스크 하고 다시" → "이제 그 기준으로 하자".
#   일 바꾸기 새 지시를 쥔 직후(grasp:3) 대신 들고 옮기는 도중(grasp:30 = 쥔 뒤 1.5초)에. 보조 = 5가지 + 쥔 블록 알아보기(--fresh-held)
#   + 가까운 빈 자리에 내려놓기(--putdown-near). 예전 기준 점수(93.8% 등)와 섞지 않는다.
#   run_np3_round.sh 의 평가 부분을 바꿈: np3 학습(3047081)은 그대로 두고 본체만 끄고 이것이 이어받는다.
#   일꾼 4개: ① np2 + 보조 28항목 (piper_np2mc) → ② np3 + 보조 28항목 (piper_np3mc) → ③ np3 단독 28항목 (piper_np3mch)
#            → ④ np2 단독 일 바꾸기 12쌍 (piper_np2mch; 혼자 하는 과제는 전환이 없어 np2h 와 같으니 다시 재지 않음)
set -u
cd "$(dirname "$0")"
export PIPER_NOPED=1 PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
SW="grasp:30"
TASKS="0 1 2 3 4 5 6 7 8 9"
PAIRS="8:0 8:3 8:5 8:7 8:9 4:5 4:9 1:7 8:1 8:4 1:8 4:1"
BEST="--approach-assist --descend-assist --unjam-assist --yaw-assist --putdown-assist --fresh-held --putdown-near"
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

while kill -0 3047081 2>/dev/null; do sleep 60; done
C2=outputs/piper_np2_model/merged
C3=outputs/piper_np3_model/merged
[ -f $C3/sufstate ] && [ -f $C3/relfeat ] || { log "np3 모델 없음 또는 relfeat 표시 없음 — np3 평가는 빼고 np2 만"; C3=""; }
log "np3 학습 끝. [새 기준 grasp:30] np2 + 보조 → np3 + 보조 → np3 단독 → np2 단독 일 바꾸기 (일꾼 4개)"
rm -rf outputs/piper/claim_m2p outputs/piper/claim_m3p outputs/piper/claim_m3h outputs/piper/claim_m2h $LOCK
worker(){ queue m2p $C2 piper_np2mc "$BEST" "$PAIRS $TASKS"
          if [ -n "$C3" ]; then queue m3p $C3 piper_np3mc "$BEST" "$PAIRS $TASKS"; queue m3h $C3 piper_np3mch "" "$PAIRS $TASKS"; fi
          queue m2h $C2 piper_np2mch "" "$PAIRS"; }
WS=""
for k in 1 2 3 4; do ( worker ) & WS="$WS $!"; sleep 30; done
wait $WS
board piper_np2mc "새 기준(옮기는 도중) np2 + 보조 (지연 9)"
[ -n "$C3" ] && board piper_np3mc "새 기준(옮기는 도중) np3 + 보조 (지연 9)" && board piper_np3mch "새 기준(옮기는 도중) np3 단독 (지연 9)"
$PY scoreboard.py --blocks --forget outputs/piper_np2h_forget --switch outputs/piper_np2mch_switch --name "새 기준(옮기는 도중) np2 단독 (혼자 과제는 np2h)" --png outputs/media/score_piper_np2mch.png --md outputs/piper/score_piper_np2mch.md | tail -1
$PY scoreboard.py --blocks --loose --forget outputs/piper_np2h_forget --switch outputs/piper_np2mch_switch --name "새 기준(옮기는 도중) np2 단독 판 안 기준" --png outputs/media/score_piper_np2mch_loose.png --md outputs/piper/score_piper_np2mch_loose.md | tail -1
for o in piper_np2mc piper_np3mc piper_np3mch piper_np2mch; do [ -d outputs/${o}_switch ] && $PY fail_classify.py outputs/$o --md outputs/piper/fail_$o.md > /dev/null; done
log "끝"
