#!/usr/bin/env bash
# 고친 보조 + ⑲ 오래 못 쥐면 쥐어 주기(--stuck-grasp 40)로 새 기준 28항목 (10/23 06:40): 고친 보조(np2fd) 28항목 93.6%·돌아오기 93.3%.
#   노란 막대 B 두 쌍 40장면 미리 보기 (run_sgd_test.sh): B 30 → 34, 셋 다 28 → 31 → 다른 항목을 해치지 않는지 28항목 전체로.
#   일꾼 1개는 4K 시연(piper-k4)이 끝나면, 3개는 piper-np2h 가 끝나면. np2 만 (np3 는 np2h 결과를 보고)
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
rm -rf outputs/piper/claim_s2p
worker(){ queue s2p $C2 piper_np2fs "$FIX" "$PAIRS $TASKS"; }
while systemctl --user is-active -q piper-k4; do sleep 30; done
( worker ) & WS="$!"
while systemctl --user is-active -q piper-np2h; do sleep 60; done
log "np2h 끝 → 일꾼 3개 더"
for k in 1 2 3; do ( worker ) & WS="$WS $!"; sleep 30; done
wait $WS
board piper_np2fs "새 기준 np2 + 고친 보조 + 오래 못 쥐면 쥐어 주기 (지연 9)"
$PY fail_classify.py outputs/piper_np2fs --md outputs/piper/fail_piper_np2fs.md > /dev/null
log "끝"
