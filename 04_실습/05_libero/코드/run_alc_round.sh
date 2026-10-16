#!/usr/bin/env bash
# 정면 카메라 가리기 학습 (10/16 16시): als1·als2 모두 손 기준 위치 입력을 쓰지 않는다 (민감도 0.1cm).
#   학습 장면의 30% 에서 정면 카메라를 가리면 (주의 마스크) 그 장면에서는 블록·판 위치를 위치 입력으로만 알 수 있다 → 쓰게 될까 (가설).
#   run_als_round_c.sh 본체(979006)는 als2 28항목 평가 중에 껐다 (평가 서브셸 4개는 그대로 돈다). 그 뒤 단계를 이것이 이어 한다.
#   ① als2 평가 서브셸이 끝나길 기다림 → 점수표
#   ② alc1 학습 (als2 에서, 같은 데이터, --cam-drop 0.3) ‖ als2 + 정렬·놓기 보조 28항목 (2개 동시)
#   ③ alc1 민감도 → alc1 28항목 (보조 평가가 끝난 자리도 가져감, 합쳐 GPU 프로세스 4개까지)
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
# 항목 하나 = 평가 한 번. 일감 이름을 mkdir 로 먼저 잡은 일꾼만 돈다 (여러 일꾼이 같은 목록을 나눠 가짐)
job(){ local q=$1 pol=$2 o=$3 opt=$4 it=$5
  mkdir outputs/piper/claim_$q/$it 2>/dev/null || return 0
  while [ "$(gpu_free)" -lt 2800 ]; do sleep 30; done      # CUDA 그래프 평가는 2.2~2.6GB
  if [ "${it#*:}" = "$it" ]; then
    $PY piper_sim/eval_piper.py --policy $pol --task-a $it --strategy none --episodes 20 --start-episode 1000 $EVQ $opt --out outputs/${o}_forget 2>&1 | grep -E "$F"
  else
    $PY piper_sim/eval_piper.py --policy $pol --task-a ${it%:*} --task-b ${it#*:} --switch-at grasp:3 --strategy keep --episodes 20 --start-episode 1000 $EVQ $opt --out outputs/${o}_switch 2>&1 | grep -E "$F"
  fi; }
queue(){ local q=$1 pol=$2 o=$3 opt=$4; mkdir -p outputs/piper/claim_$q
  for it in $PAIRS $TASKS; do job $q $pol $o "$opt" $it; done; }   # 오래 걸리는 쌍 먼저

# ① als2 평가 서브셸 (본체가 띄운 것)
for p in 1322491 1322492 1322493 1322497; do while kill -0 $p 2>/dev/null; do sleep 60; done; done
log "als2 28항목 끝"
board piper_als2h "상태를 동작 쪽에도 als2 (새 배치 20장면, 지연 9)"

# ② alc1 학습 ‖ als2 + 보조 28항목 2개
A2=outputs/piper_als2_model/merged
rm -rf outputs/piper/claim_as2 outputs/piper/claim_alc1
log "[alc1 학습] als2 에서, 정면 카메라 가리기 30% ‖ [als2 + 정렬·놓기 보조 28항목] 2개 동시"
( $PY train_lora.py --policy $A2 --data $DATA $TR --cam-drop 0.3 --dagger-frac 0.35 --batch-size 8 --grad-accum 2 --steps 10000 --lr 3e-5 \
    --eval-every 2500 --save-every 5000 --out outputs/piper_alc1_model 2>&1 | grep --line-buffered -E '에피소드 학습|상태를 동작|카메라 가리기|val_loss|모델 저장|Traceback|Error' ) & T=$!
sleep 300     # 학습이 GPU 메모리를 먼저 잡게
( queue as2 $A2 piper_als2place "--place-assist" ) & W1=$!
sleep 120
( queue as2 $A2 piper_als2place "--place-assist" ) & W2=$!
wait $T
C1=outputs/piper_alc1_model/merged
[ -f $C1/sufstate ] || { log "alc1 모델 없음 — 멈춤"; exit 1; }

# ③ 민감도 (CPU) → alc1 28항목: 새 일꾼 2개 + 보조 평가를 마친 일꾼이 넘어옴
log "[alc1 민감도]"
CUDA_VISIBLE_DEVICES= nice -n 10 $PY sens_check.py $C1 2>&1 | grep -E "목표 위치|Error|Traceback"
log "[alc1 28항목]"
( queue alc1 $C1 piper_alc1h "" ) & W3=$!
sleep 120
( queue alc1 $C1 piper_alc1h "" ) & W4=$!
wait $W1; ( queue alc1 $C1 piper_alc1h "" ) & W5=$!
wait $W2; ( queue alc1 $C1 piper_alc1h "" ) & W6=$!
wait $W3 $W4 $W5 $W6
board piper_als2place "als2 + 정렬·놓기 보조 (지연 9)"
board piper_alc1h "정면 카메라 가리기 alc1 (지연 9)"
log "끝"
