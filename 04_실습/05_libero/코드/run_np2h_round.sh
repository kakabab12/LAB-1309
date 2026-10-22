#!/usr/bin/env bash
# 판 따라가기(⑯)를 뺀 고침으로 새 기준 28항목 (10/23 03:35): run_np2g_round.sh 에서 A4→B5→A4 돌아오기가 17 → 8/20 으로 무너짐.
#   실패 12장면 모두 회색 판이 그대로인데 ⑯ 이 판 기억을 7~29번 고침 (팔이 판을 가려 잘못 잼) → ⑤ 가 엉뚱한 자리 기준으로 놓기를 막아 빨강을 끝까지 들고 있음.
#   설정 = 넓은 변환식 + ⑰ 기억 고치기 + ⑮' 먼 빈 자리 (⑩·⑮·⑯ 뺌). 일꾼 3개 + 막힌 쥐기 시험(piper-sgtest)이 끝나면 1개 더
#   ① np2 + 고침 28항목 (piper_np2fd) → ② np3 + 고침 28항목 (piper_np3fd) → ③ np3 단독 28항목 (piper_np3mch) → ④ np2 단독 일 바꾸기 12쌍 (piper_np2mch)
#   일꾼 3개 + 막힌 쥐기 시험이 끝나면 1개
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
FIX="--approach-assist --descend-assist --unjam-assist --yaw-assist --putdown-assist --fresh-held --putdown-near --wrong-guard --mem-refresh --putdown-far"
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
log "[새 기준 + 고침(⑯ 뺌)] np2 → np3, 그다음 단독 (일꾼 3개 + 시험 끝나면 1개)"
rm -rf outputs/piper/claim_d2p outputs/piper/claim_d3p outputs/piper/claim_h3 outputs/piper/claim_h2
worker(){ queue d2p $C2 piper_np2fd "$FIX" "$PAIRS $TASKS"
          queue d3p $C3 piper_np3fd "$FIX" "$PAIRS $TASKS"
          queue h3 $C3 piper_np3mch "" "$PAIRS $TASKS"
          queue h2 $C2 piper_np2mch "" "$PAIRS"; }
WS=""
for k in 1 2 3; do ( worker ) & WS="$WS $!"; sleep 30; done
while systemctl --user is-active -q piper-sgtest; do sleep 60; done
( worker ) & WS="$WS $!"
wait $WS
board piper_np2fd "새 기준 np2 + 고친 보조 (지연 9)"
board piper_np3fd "새 기준 np3 + 고친 보조 (지연 9)"
board piper_np3mch "새 기준 np3 단독 (지연 9)"
$PY scoreboard.py --blocks --forget outputs/piper_np2h_forget --switch outputs/piper_np2mch_switch --name "새 기준 np2 단독 (혼자 과제는 np2h)" --png outputs/media/score_piper_np2mch.png --md outputs/piper/score_piper_np2mch.md | tail -1
$PY scoreboard.py --blocks --loose --forget outputs/piper_np2h_forget --switch outputs/piper_np2mch_switch --name "새 기준 np2 단독 판 안 기준" --png outputs/media/score_piper_np2mch_loose.png --md outputs/piper/score_piper_np2mch_loose.md | tail -1
for o in piper_np2fd piper_np3fd piper_np3mch piper_np2mch; do [ -d outputs/${o}_switch ] && $PY fail_classify.py outputs/$o --md outputs/piper/fail_$o.md > /dev/null; done
log "끝"
