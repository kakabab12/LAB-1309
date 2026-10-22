#!/usr/bin/env bash
# 보조 증류 np3 (10/22 11시): VLA + 보조 93.8% 인데 VLA 혼자 25.5%. 보조가 고친 동작을 VLA 가 배우게 한다.
#   ① 보조 시범 모으기 (assist_collect.py): np2 + 보조 5가지로 평가와 똑같이, 학습용 장면 3300~3319 × 22개 (일꾼 4개).
#      성공한 구간만 저장 (혼자 과제 A, 일 바꾸기 B, 돌아오기 A2) → data/piper_np_assist1 (파일 이름 DA… = 교정 시범 몫)
#   ② 정리 (trim_stalls·물체 위치·손 기준 위치·사진 특징) → ③ np3 학습 (np2 에서, 지금 데이터 + 보조 시범, 10,000스텝)
#   ④ 일꾼 4개: np3 + 보조 5가지 28항목 (piper_np3pd) → np3 단독 28항목 (piper_np3h)
#   비교: np2 + 보조 5가지 93.8 / 94.6 (piper_np2pd), np2 단독 25.5 / 36.4 (piper_np2h)
set -u
cd "$(dirname "$0")"
export PIPER_NOPED=1 PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --log-every 500 --abs-pos --objfeat --feat relfeat --vis-cache cache/vis --frame-stride 3 --sufstate"
TASKS="0 1 2 3 4 5 6 7 8 9"
PAIRS="8:0 8:3 8:5 8:7 8:9 4:5 4:9 1:7 8:1 8:4 1:8 4:1"
PD="--approach-assist --descend-assist --unjam-assist --yaw-assist --putdown-assist"
NPD="data/piper_np_normal_t data/piper_np_wide_normal_t data/piper_np_switch_t data/piper_np_recovery_t"
ASD=data/piper_np_assist1
B=outputs/piper_np2_model/merged
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
# 일 하나 = 22개 중 하나. kind=eval 이면 eval_piper, collect 면 assist_collect (같은 인자 + --data-out)
job(){ local q=$1 pol=$2 o=$3 opt=$4 it=$5 kind=$6 start=$7
  mkdir outputs/piper/claim_$q/$it 2>/dev/null || return 0
  until mkdir $LOCK 2>/dev/null; do sleep 15; done
  while [ "$(gpu_free)" -lt 2800 ]; do sleep 30; done
  local prog="piper_sim/eval_piper.py" pre=""
  [ $kind = collect ] && prog="assist_collect.py" && pre="--data-out $ASD --"
  if [ "${it#*:}" = "$it" ]; then
    ( $PY $prog $pre --policy $pol --task-a $it --strategy none --episodes 20 --start-episode $start $EVQ $opt --out outputs/${o}_forget 2>&1 | grep -E "$F" ) & local E=$!
  else
    ( $PY $prog $pre --policy $pol --task-a ${it%:*} --task-b ${it#*:} --switch-at grasp:3 --strategy keep --episodes 20 --start-episode $start $EVQ $opt --out outputs/${o}_switch 2>&1 | grep -E "$F" ) & local E=$!
  fi
  sleep 150; rmdir $LOCK
  wait $E; }
queue(){ local q=$1 pol=$2 o=$3 opt=$4 kind=$5 start=$6; mkdir -p outputs/piper/claim_$q
  for it in $PAIRS $TASKS; do job $q $pol $o "$opt" $it $kind $start; done; }

# ① 보조 시범 모으기
log "[보조 시범] np2 + 보조 5가지, 학습용 장면 3300~3319 × 22개 (일꾼 4개)"
rm -rf outputs/piper/claim_n3c $LOCK
WS=""
for k in 1 2 3 4; do ( queue n3c $B piper_np2_assistcol "$PD" collect 3300 ) & WS="$WS $!"; sleep 30; done
wait $WS
log "보조 시범 $(ls $ASD/episodes | wc -l)개"

# ② 정리 → ③ np3 학습
prep_data $ASD $B
DATA="$NPD data/piper_np_dagger_np1_t data/piper_np_switch2_t ${ASD}_t"
log "[np3 학습] np2 에서, + 보조 시범, 10000스텝"
$PY train_lora.py --policy $B --data $DATA $TR --dagger-frac 0.35 --batch-size 8 --grad-accum 2 --steps 10000 --lr 3e-5 \
    --eval-every 2500 --save-every 5000 --exclude outputs/piper/exclude_rise_np.txt --out outputs/piper_np3_model 2>&1 \
    | grep --line-buffered -E '에피소드 학습|상태를 동작|교정 시범 비율|val_loss|모델 저장|Traceback|Error'
C=outputs/piper_np3_model/merged
[ -f $C/sufstate ] && [ -f $C/relfeat ] || { log "np3 모델 없음 또는 relfeat 표시 없음 — 멈춤"; exit 1; }

# ④ 평가
log "[np3 + 보조 5가지 28항목] → [np3 단독 28항목] (일꾼 4개)"
rm -rf outputs/piper/claim_n3p outputs/piper/claim_n3h $LOCK
worker(){ queue n3p $C piper_np3pd "$PD" eval 1000; queue n3h $C piper_np3h "" eval 1000; }
WS=""
for k in 1 2 3 4; do ( worker ) & WS="$WS $!"; sleep 30; done
wait $WS
board piper_np3pd "받침대 없음 np3 + 보조 5가지 (지연 9)"
board piper_np3h "받침대 없음 np3 단독 (지연 9)"
for o in piper_np3pd piper_np3h; do $PY fail_classify.py outputs/$o --md outputs/piper/fail_$o.md > /dev/null; done
log "끝"
