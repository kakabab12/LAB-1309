#!/usr/bin/env bash
# 4K 시연 다시 (10/23 07시): 실행기 끝 판정이 쌓기를 몰라(책상 식으로 위·아래 7cm) 쌓은 뒤 맴돎 → front_calib_stack.json 으로 고침.
#   np2s 일꾼(3635022)을 멈춰 비운 자리에서 시연을 돌린 뒤, np2s 의 s2p 줄(고친 보조 + ⑲ 28항목)을 이어받아 돈다 (자리표가 mkdir 라 같이 써도 됨)
set -u
cd "$(dirname "$0")"
export PIPER_NOPED=1 PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl PIPER_FRONT_CALIB=piper_sim/front_calib_wide.json
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph --release-wait 40"
SW="grasp:30"
TASKS="0 1 2 3 4 5 6 7 8 9"
PAIRS="8:0 8:3 8:5 8:7 8:9 4:5 4:9 1:7 8:1 8:4 1:8 4:1"
FIX="--approach-assist --descend-assist --unjam-assist --yaw-assist --putdown-assist --fresh-held --putdown-near --wrong-guard --mem-refresh --putdown-far --stuck-grasp 40"
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
log "[새 기준 + 고친 보조 + ⑲] np2 28항목 (일꾼 1개 먼저, np2h 끝나면 3개 더)"
while kill -0 3635034 2>/dev/null; do sleep 20; done
until mkdir $LOCK 2>/dev/null; do sleep 10; done
( sleep 90; rmdir $LOCK ) &
rm -f outputs/media/k4_fix_1004.mp4
timeout 2400 $PY korean_live.py --episode 1004 --script "0=빨간 블록을 회색 판에 올려줘" "grasp+30=잠깐, 초록 블록 먼저 파란 블록 위에 올리고 하던 거 이어서 해줘" \
  --render4k outputs/media/k4_fix_1004 --title "PiPER 시뮬 · 한국어 명령 · 옮기는 도중 끼어들기 (np2 + 고친 보조)"
wait
log "4K 시연 끝 → s2p 줄 이어받음"
queue s2p $C2 piper_np2fs "$FIX" "$PAIRS $TASKS"
log "끝"
