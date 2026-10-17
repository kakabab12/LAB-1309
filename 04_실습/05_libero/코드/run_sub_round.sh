#!/usr/bin/env bash
# 보조를 켠 채 교정 시범 (10/17 12시): als2 + 정렬·놓기·내려가기·다가가기 (28항목 61.4%) 의 남은 실패 대부분이
#   '블록 위에서 못 쥠' — 쥐는 높이까지 내려와도 학생이 오므리지 않고 손을 올린다. 쥐기를 규칙으로 해 줘도(⑩) 그 뒤 들어 올리지 않는다.
#   → 학생이 보조와 함께 실제로 가는 상태에서 시범 프로그램이 넘겨받은 동작을 배우게 한다 (배치 조건 = 보조 켬).
#   ① 손목 맞추기 시험(run_yaw_test.sh 2079248) 끝 → ② als2 교정 시범 (보조 켬, 항목은 als2apr 성적으로) → ③ 정리
#   ④ als3a 학습 (als2 에서) → ⑤ 민감도 → ⑥ als3a + 보조 28항목 → als3a 단독 28항목
# run_sub_round.sh (10/17 15시): 사용자와 정한 '단계 나누기' — als3a 학습(2284901)은 그대로 두고 본체(run_asd_round.sh)는 끔.
#   ① als3a 학습 끝 → ② sub1 학습 (als2 에서, 같은 데이터 + 보조 켠 교정 시범, 지시문 대신 단계 지시 subtask.py --subtask)
#   ③ 일꾼 4개: als3a + 보조 → sub1 + 보조 → sub1 (단계 지시만) → als3a 단독, 28항목씩 (sub1 은 평가에서 단계 지시 자동)
set -u
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
DGQ="--latency-steps 9 --ttrtc --n-action-steps 1"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --log-every 500 --abs-pos --objfeat --feat relfeat --vis-cache cache/vis --frame-stride 3 --sufstate"
DATA="data/piper_al_normal_t data/piper_al_wide_normal_t data/piper_al_switch_t data/piper_al_recovery_t data/piper_al_dagger_bo1_t data/piper_al_dagger_al1_t data/piper_al_dagger_als1_t"
TASKS="0 1 2 3 4 5 6 7 8 9"
PAIRS="8:0 8:3 8:5 8:7 8:9 4:5 4:9 1:7 8:1 8:4 1:8 4:1"
AS="--approach-assist --descend-assist"
prep_data(){ local D=$1 B=$2
  rm -rf ${D}_t; $PY piper_sim/trim_stalls.py $D ${D}_t
  for s in 0 1 2; do ( $PY objfeat_cache.py --data ${D}_t --shard $s --nshard 3 > /dev/null 2>&1 ) & done; wait
  $PY relfeat.py --data ${D}_t
  ( $PY vis_cache.py build --policy $B --data ${D}_t --out cache/vis --batch 8 --stride 3 --shard 0 --nshard 2 2>&1 | grep -E "끝|Error|Traceback" ) & local Q1=$!
  ( $PY vis_cache.py build --policy $B --data ${D}_t --out cache/vis --batch 8 --stride 3 --shard 1 --nshard 2 2>&1 | grep -E "끝|Error|Traceback" ) & local Q2=$!
  wait $Q1 $Q2; }
train(){ local init=$1 out=$2 steps=$3 lr=$4 frac=$5; shift 5
  $PY train_lora.py --policy $init --data $DATA "$@" $TR --dagger-frac $frac --batch-size 8 --grad-accum 2 --steps $steps --lr $lr \
    --eval-every 2500 --save-every 5000 --out $out 2>&1 | grep --line-buffered -E '에피소드 학습|상태를 동작|val_loss|모델 저장|Traceback|Error'; }

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


DATA="$DATA data/piper_al_dagger_als2a_t"
while kill -0 2284901 2>/dev/null; do sleep 60; done
[ -f outputs/piper_als3a_model/merged/sufstate ] || { log "als3a 모델 없음 — 멈춤"; exit 1; }
log "als3a 학습 끝"
B=outputs/piper_als2_model/merged
log "[sub1 학습] als2 에서, 단계 지시 (pick up / put it on / put it down)"
train $B outputs/piper_sub1_model 10000 3e-5 0.35 --subtask
S=outputs/piper_sub1_model/merged
[ -f $S/subtask ] || { log "sub1 모델 없음 — 멈춤"; exit 1; }
C=outputs/piper_als3a_model/merged
log "[als3a + 보조] → [sub1 + 보조] → [sub1 단계 지시만] → [als3a 단독] 28항목씩 (일꾼 4개)"
rm -rf outputs/piper/claim_a3p outputs/piper/claim_s1p outputs/piper/claim_s1h outputs/piper/claim_a3h $LOCK
worker(){ queue a3p $C piper_als3aapr "$AS"; queue s1p $S piper_sub1apr "$AS"; queue s1h $S piper_sub1h ""; queue a3h $C piper_als3ah ""; }
WS=""
for k in 1 2 3 4; do ( worker ) & WS="$WS $!"; sleep 30; done
wait $WS
board piper_als3aapr "als3a + 정렬·놓기·내려가기·다가가기 (지연 9)"
board piper_sub1apr "sub1 단계 지시 + 정렬·놓기·내려가기·다가가기 (지연 9)"
board piper_sub1h "sub1 단계 지시만 (지연 9)"
board piper_als3ah "als3a 단독 (지연 9)"
for o in piper_als3aapr piper_sub1apr piper_sub1h piper_als3ah; do PIPER_BLOCKS=1 $PY fail_classify.py outputs/$o --md outputs/piper/fail_$o.md > /dev/null; done
log "끝"
