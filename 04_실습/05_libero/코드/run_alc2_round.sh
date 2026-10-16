#!/usr/bin/env bash
# 새 층 학습률 키우기 (10/16 23시): als1·als2·alc1 모두 손 기준 위치 입력을 쓰지 않는다. alc1 은 정면 카메라를 가려도 그대로.
#   가중치를 보니 상태를 동작 토큰에 더하는 새 층(state_suffix_proj, 0 에서 시작)이 3e-5 로는 10,000스텝에 원소당 0.003 수준.
#   → 그 층만 1e-3 (--new-lr), 정면 카메라 가리기 30% 는 그대로. alc1 에서 이어 학습 = alc2.
#   run_alc_round_d.sh 는 끔: 돌던 alc1 평가 4개(쌍 8:0·8:3·8:5·8:7)만 끝내고 alc1 의 나머지 항목은 재지 않는다 (민감도가 그대로라).
#   ① 돌던 평가 4개 끝 → ② alc2 학습 (GPU 혼자, 2500스텝마다 저장) ‖ 2500스텝 저장본 민감도 (CPU)
#   ③ alc2 민감도 → 일꾼 4개: alc2 28항목 → als2 + 정렬·놓기 보조 28항목
set -u
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --log-every 500 --abs-pos --objfeat --feat relfeat --vis-cache cache/vis --frame-stride 3 --sufstate"
DATA="data/piper_al_normal_t data/piper_al_wide_normal_t data/piper_al_switch_t data/piper_al_recovery_t data/piper_al_dagger_bo1_t data/piper_al_dagger_al1_t data/piper_al_dagger_als1_t"
TASKS="0 1 2 3 4 5 6 7 8 9"
PAIRS="8:0 8:3 8:5 8:7 8:9 4:5 4:9 1:7 8:1 8:4 1:8 4:1"
board(){ $PY scoreboard.py --blocks --forget outputs/$1_forget --switch outputs/$1_switch --name "$2" --png outputs/media/score_$1.png --md outputs/piper/score_$1.md | tail -1
         $PY scoreboard.py --blocks --loose --forget outputs/$1_forget --switch outputs/$1_switch --name "$2 판 안 기준" --png outputs/media/score_$1_loose.png --md outputs/piper/score_$1_loose.md | tail -1; }
gpu_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
LOCK=outputs/piper/gpu_launch.lock
# 항목 하나 = 평가 한 번. mkdir 로 먼저 잡은 일꾼만 돈다. 띄우기는 자물쇠 + 남은 GPU 메모리 2.8GB, 띄운 뒤 2분 반 기다림
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
sens(){ CUDA_VISIBLE_DEVICES= nice -n 10 $PY sens_check.py $1 2>&1 | grep -E "목표 위치|Error|Traceback" | sed 's/손 \[[^]]*\], //'; }

# ① 돌던 alc1 평가 4개
for p in 1529303 1530843 1532357 1533862; do while kill -0 $p 2>/dev/null; do sleep 30; done; done
rm -rf outputs/piper/claim_as2 outputs/piper/claim_alc2 $LOCK
log "alc1 평가 4개 끝"

# ② alc2 학습 (alc1 에서, 새 층 1e-3, 정면 가리기 30%) ‖ 2500스텝 저장본 민감도
M=outputs/piper_alc2_model
log "[alc2 학습] alc1 에서, 새 층 학습률 1e-3, 정면 카메라 가리기 30%"
( $PY train_lora.py --policy outputs/piper_alc1_model/merged --data $DATA $TR --cam-drop 0.3 --new-lr 1e-3 --dagger-frac 0.35 \
    --batch-size 8 --grad-accum 2 --steps 10000 --lr 3e-5 --eval-every 2500 --save-every 2500 --out $M 2>&1 \
    | grep --line-buffered -E '에피소드 학습|상태를 동작|카메라 가리기|새 층|val_loss|모델 저장|Traceback|Error' ) & T=$!
( until [ -f $M/step_2500/sufstate ]; do sleep 60; done; sleep 60
  log "[alc2 2500스텝 민감도]"; sens $M/step_2500 ) & S=$!
wait $T $S
C2=$M/merged
[ -f $C2/sufstate ] || { log "alc2 모델 없음 — 멈춤"; exit 1; }
log "[alc2 민감도]"
sens $C2

# ③ 일꾼 4개: alc2 28항목 → als2 + 정렬·놓기 보조 28항목
A2=outputs/piper_als2_model/merged
log "[alc2 28항목] → [als2 + 정렬·놓기 보조 28항목] (일꾼 4개)"
worker(){ queue alc2 $C2 piper_alc2h ""; queue as2 $A2 piper_als2place "--place-assist"; }
WS=""
for k in 1 2 3 4; do ( worker ) & WS="$WS $!"; sleep 30; done
wait $WS
board piper_alc2h "새 층 학습률 키움 alc2 (지연 9)"
board piper_als2place "als2 + 정렬·놓기 보조 (지연 9)"
log "끝"
