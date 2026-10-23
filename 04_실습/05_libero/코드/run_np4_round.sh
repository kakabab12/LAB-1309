#!/usr/bin/env bash
# 보조 증류 np4 (10/23 18:30): 사용자 "셋 다 성공도 90 넘기자". np3 + 새 설정 셋 다 86~88% 에서 보조를 하나씩 고쳐도 ±1~3장면이라,
#   np2 → np3 때처럼 (셋 다 78 → 87.5) 보조가 고친 동작을 VLA 가 배우게 한다. 이번에는 새 기준(쥔 뒤 1.5초 일 바꾸기, 빈 자리에 내려놓기, 돌아오기).
#   ① 보조 시범 (assist_collect.py): np3 + 새 설정 + ㉒' + ㉑, 새 기준 그대로, 학습용 장면 3320~3339 × 22개 (일꾼 4개) → data/piper_np_assist2
#   ② 정리 → ③ np4 학습 (np3 에서, 지금 데이터 + 보조 시범 1·2, 10,000스텝)
#   ④ np4 + 새 설정 28항목 (piper_np4x), 돌아오기 6쌍 먼저 (일꾼 4개)
#   비교: np3 + 새 설정 (piper_np3x) 28항목 93.4%, 돌아오기 92.5%, 셋 다 86.7%
set -u
cd "$(dirname "$0")"
export PIPER_NOPED=1 PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl PIPER_FRONT_CALIB=piper_sim/front_calib_wide.json
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph --release-wait 40"
SW="grasp:30"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --log-every 500 --abs-pos --objfeat --feat relfeat --vis-cache cache/vis --frame-stride 3 --sufstate"
TASKS="0 1 2 3 4 5 6 7 8 9"
PAIRS="8:5 8:7 8:9 4:5 4:9 1:7 8:0 8:3 8:1 8:4 1:8 4:1"
BEST="--approach-assist --descend-assist --unjam-assist --yaw-assist --putdown-assist --fresh-held --putdown-near --wrong-guard --mem-refresh --putdown-far --stuck-grasp 40 --yaw-none-reset 10 --mem-tol 0.012 --open-empty --stack-guard --stack-clear --guard-r 0.07"
NPD="data/piper_np_normal_t data/piper_np_wide_normal_t data/piper_np_switch_t data/piper_np_recovery_t"
ASD=data/piper_np_assist2
B=outputs/piper_np3_model/merged
board(){ $PY scoreboard.py --blocks --forget outputs/$1_forget --switch outputs/$1_switch --name "$2" --png outputs/media/score_$1.png --md outputs/piper/score_$1.md | tail -1
         $PY scoreboard.py --blocks --loose --forget outputs/$1_forget --switch outputs/$1_switch --name "$2 판 안 기준" --png outputs/media/score_$1_loose.png --md outputs/piper/score_$1_loose.md | tail -1; }
gpu_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
LOCK=outputs/piper/gpu_launch.lock
prep_data(){ local D=$1 B=$2
  rm -rf ${D}_t; $PY piper_sim/trim_stalls.py $D ${D}_t
  for s in 0 1 2; do ( $PY objfeat_cache.py --data ${D}_t --shard $s --nshard 3 > /dev/null 2>&1 ) & done; wait
  $PY relfeat.py --data ${D}_t
  ( $PY vis_cache.py build --policy $B --data ${D}_t --out cache/vis --batch 8 --stride 3 --shard 0 --nshard 2 2>&1 | grep -E "끝|Error|Traceback" ) & local Q1=$!
  ( $PY vis_cache.py build --policy $B --data ${D}_t --out cache/vis --batch 8 --stride 3 --shard 1 --nshard 2 2>&1 | grep -E "끝|Error|Traceback" ) & local Q2=$!
  wait $Q1 $Q2; }
job(){ local q=$1 pol=$2 o=$3 opt=$4 it=$5 kind=$6 start=$7
  mkdir outputs/piper/claim_$q/$it 2>/dev/null || return 0
  until mkdir $LOCK 2>/dev/null; do sleep 15; done
  while [ "$(gpu_free)" -lt 2800 ]; do sleep 30; done
  local prog="piper_sim/eval_piper.py" pre=""
  [ $kind = collect ] && prog="assist_collect.py" && pre="--data-out $ASD --"
  if [ "${it#*:}" = "$it" ]; then
    ( $PY $prog $pre --policy $pol --task-a $it --strategy none --episodes 20 --start-episode $start $EVQ $opt --out outputs/${o}_forget 2>&1 | grep -E "$F" ) & local E=$!
  else
    ( $PY $prog $pre --policy $pol --task-a ${it%:*} --task-b ${it#*:} --switch-at $SW --strategy keep --episodes 20 --start-episode $start $EVQ $opt --out outputs/${o}_switch 2>&1 | grep -E "$F" ) & local E=$!
  fi
  sleep 150; rmdir $LOCK
  wait $E; }
queue(){ local q=$1 pol=$2 o=$3 opt=$4 kind=$5 start=$6; mkdir -p outputs/piper/claim_$q
  for it in $PAIRS $TASKS; do job $q $pol $o "$opt" $it $kind $start; done; }

# ① 보조 시범 모으기
log "[보조 시범] np3 + 새 설정, 새 기준, 학습용 장면 3320~3339 × 22개 (일꾼 4개)"
rm -rf outputs/piper/claim_n4c
WS=""
for k in 1 2 3 4; do ( queue n4c $B piper_np3_assistcol2 "$BEST" collect 3320 ) & WS="$WS $!"; sleep 30; done
wait $WS
log "보조 시범 $(ls $ASD/episodes | wc -l)개"

# ② 정리 → ③ np4 학습
prep_data $ASD $B
DATA="$NPD data/piper_np_dagger_np1_t data/piper_np_switch2_t data/piper_np_assist1_t ${ASD}_t"
log "[np4 학습] np3 에서, + 보조 시범 1·2, 10000스텝"
$PY train_lora.py --policy $B --data $DATA $TR --dagger-frac 0.35 --batch-size 8 --grad-accum 2 --steps 10000 --lr 3e-5 \
    --eval-every 2500 --save-every 5000 --exclude outputs/piper/exclude_rise_np.txt --out outputs/piper_np4_model 2>&1 \
    | grep --line-buffered -E '에피소드 학습|상태를 동작|교정 시범 비율|val_loss|모델 저장|Traceback|Error'
C=outputs/piper_np4_model/merged
[ -f $C/sufstate ] && [ -f $C/relfeat ] || { log "np4 모델 없음 또는 relfeat 표시 없음 — 멈춤"; exit 1; }

# ④ 평가
log "[np4 + 새 설정 28항목] 돌아오기 6쌍 먼저 (일꾼 4개)"
rm -rf outputs/piper/claim_n4p
WS=""
for k in 1 2 3 4; do ( queue n4p $C piper_np4x "$BEST" eval 1000 ) & WS="$WS $!"; sleep 30; done
wait $WS
board piper_np4x "새 기준 np4 + 새 설정 (지연 9)"
$PY fail_classify.py outputs/piper_np4x --md outputs/piper/fail_piper_np4x.md > /dev/null
log "끝"
