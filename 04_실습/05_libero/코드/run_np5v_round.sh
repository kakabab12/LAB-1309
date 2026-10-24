#!/usr/bin/env bash
# VLA 단독 올리기 (10/24, 사용자 "보조 없이 VLA 정확도 올리는 방법 찾아줘")
#   ① 학습이 반올림에 지워짐: 동작 전문가 가중치 257개가 bfloat16 이라 3e-5 학습률의 갱신이 거의 다 반올림에 지워졌다
#      (np2→np3 10000스텝에 바뀐 값 32%, |w|≥0.03 은 0%; EMA 0.999 도 bfloat16 이라 후반 갱신이 안 쌓임) → --fp32-master
#   ② VLM 32층(사진·지시문·상태 토큰을 읽는 쪽)은 LIBERO 체크포인트 그대로 얼려 둠 → --vlm-lora 16
#   np3 단독 혼자 과제 실패 131 중 79 가 놓기 (놓는 순간 손이 판 가운데에서 3.5~4.4cm, 손은 멈춰 있음, 방향은 흩어짐 = 위치를 흐릿하게 앎)
#   np5f = np3 + ① (같은 데이터 10000스텝), np5v = np3 + ① + ②. 단독 혼자 10 → 돌아오기 6쌍 → 나머지 6쌍 (새 기준, 지연 9). 비교 = np3 단독 piper_np3mch
cd "$(dirname "$0")"
export PIPER_NOPED=1 PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl PIPER_FRONT_CALIB=piper_sim/front_calib_wide.json
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph --release-wait 40"
SW="grasp:30"
TASKS="0 1 2 3 4 5 6 7 8 9"
RES="8:5 8:7 8:9 4:5 4:9 1:7"
OTH="8:0 8:3 8:1 8:4 1:8 4:1"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --log-every 250 --abs-pos --objfeat --feat relfeat --vis-cache cache/vis --frame-stride 3 --sufstate"
NPD="data/piper_np_normal_t data/piper_np_wide_normal_t data/piper_np_switch_t data/piper_np_recovery_t"
DATA="$NPD data/piper_np_dagger_np1_t data/piper_np_switch2_t data/piper_np_assist1_t"
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
train(){ local out=$1; shift
  log "[학습 $out] np3 에서 10000스텝 $* (MemAvailable $(grep MemAvailable /proc/meminfo | awk '{print int($2/1e6)}')GB)"
  $PY train_lora.py --policy outputs/piper_np3_model/merged --data $DATA $TR --dagger-frac 0.35 --batch-size 8 --grad-accum 2 --steps 10000 --lr 3e-5 \
      "$@" --eval-every 2500 --save-every 5000 --exclude outputs/piper/exclude_rise_np.txt --out $out 2>&1 \
      | grep --line-buffered -E '에피소드 학습|VLM LoRA|float32 원본|"step": [0-9]*000, "loss"|val_loss|모델 저장|정책 저장|Traceback|Error|Killed'
  [ -f $out/merged/config.json ] || { log "학습 실패 ($out) — 멈춤"; exit 1; }; }
evals(){ local C=$1 o=$2 q=$3 items=$4
  rm -rf outputs/piper/claim_$q
  WS=""
  for k in 1 2 3 4; do ( queue $q $C $o "" "$items" ) & WS="$WS $!"; sleep 30; done
  wait $WS; }

log "[np5] piper-alone (np2 단독)·탐침·시험 끝나기 기다림"
while systemctl --user is-active --quiet piper-alone || systemctl --user is-active --quiet probe2 || systemctl --user is-active --quiet vlmlora-smoke; do sleep 60; done
train outputs/piper_np5f_model --fp32-master
log "[np5f 단독] 혼자 10 (일꾼 4개)"
evals outputs/piper_np5f_model/merged piper_np5fh f5 "$TASKS"
.venv/bin/python fail_classify.py outputs/piper_np5fh | grep "혼자 | \*\*합계"
train outputs/piper_np5v_model --fp32-master --vlm-lora 16 --vlm-lora-lr 2e-4
log "[np5v 단독] 혼자 10 (일꾼 4개)"
evals outputs/piper_np5v_model/merged piper_np5vh v5 "$TASKS"
.venv/bin/python fail_classify.py outputs/piper_np5vh | grep "혼자 | \*\*합계"
log "[돌아오기 6쌍 → 나머지 6쌍] np5f·np5v 단독"
rm -rf outputs/piper/claim_f5p outputs/piper/claim_v5p
WS=""
for k in 1 2 3 4; do ( queue v5p outputs/piper_np5v_model/merged piper_np5vh "" "$RES"; queue f5p outputs/piper_np5f_model/merged piper_np5fh "" "$RES"
                       queue v5p outputs/piper_np5v_model/merged piper_np5vh "" "$OTH"; queue f5p outputs/piper_np5f_model/merged piper_np5fh "" "$OTH" ) & WS="$WS $!"; sleep 30; done
wait $WS
board piper_np5fh "새 기준 np5f 단독 (float32 원본, 지연 9)"
board piper_np5vh "새 기준 np5v 단독 (float32 원본 + VLM LoRA, 지연 9)"
log "끝"
