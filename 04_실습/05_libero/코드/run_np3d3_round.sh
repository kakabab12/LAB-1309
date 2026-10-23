#!/usr/bin/env bash
# np3 + 새 설정 + ㉒' + ㉓ 28항목 나머지 (10/24 03:40): 돌아오기 6쌍 셋 다 240장면 217 (90.4%) — 28항목 점수를 내려고 나머지 6쌍 + 혼자 10.
#   결과는 같은 폴더 outputs/piper_np3d (6쌍은 run_np3d_round.sh)
#   일꾼 4개. 6쌍만 (28항목은 결과를 보고)
set -u
cd "$(dirname "$0")"
export PIPER_NOPED=1 PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl PIPER_FRONT_CALIB=piper_sim/front_calib_wide.json
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph --release-wait 40"
SW="grasp:30"
TASKS="0 1 2 3 4 5 6 7 8 9"
PAIRS="8:0 8:3 8:1 8:4 1:8 4:1"
FIX="--approach-assist --descend-assist --unjam-assist --yaw-assist --putdown-assist --fresh-held --putdown-near --wrong-guard --mem-refresh --putdown-far --stuck-grasp 40 --yaw-none-reset 10 --mem-tol 0.012 --open-empty --stack-guard --stack-clear --guard-r 0.07 --deliver"
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
rm -rf outputs/piper/claim_d6p
log "[np3 + 새 설정 + ㉒' + ㉓] 돌아오기 6쌍 (일꾼 4개)"
worker(){ queue d6p $C3 piper_np3d "$FIX" "$PAIRS $TASKS"; }
WS=""
for k in 1 2 3 4; do ( worker ) & WS="$WS $!"; sleep 30; done
wait $WS
board piper_np3d "새 기준 np3 + 고친 보조 전부 (⑲⑨'⑰'⑳㉒'㉓, 지연 9)"
$PY fail_classify.py outputs/piper_np3d --md outputs/piper/fail_piper_np3d.md > /dev/null
log "끝"
