#!/usr/bin/env bash
# sub2 (10/19 23시, 사용자 "내일 아침 9시까지만"): 원래 지시문을 살린 단계 지시 (subtask.py --style tag).
#   sub1 (단계만, 놓을 곳 없음) + 보조 48.2% — 집기 단계가 나빠짐 (전환 안 됨 52/240). 놓을 곳을 빼서 쥐는 방향이 여러 개가 됐다는 가설.
#   ① sub2 학습 (als2 에서, sub1·als3a 와 같은 데이터, 10,000스텝) → ② sub2 + 보조 28항목 (일꾼 4개) → 점수표·실패 나누기
set -u
# run_sub2_eval.sh (10/20 03시): 쥔 블록 판정을 지금 사진으로 고친 뒤 sub2 + 보조 28항목을 처음부터 (예전 판정 3항목은 outputs/old_sub2apr_before_fresh)
# (이전 메모) 10/20 02시: 일꾼 하나를 껐다 (단계 지시 길이마다 CUDA 그래프가 늘어 프로세스가 2.9GB 까지 커짐). 예전 코드로 도는 평가 하나가 끝나면 이 일꾼이 대신 들어간다 (새 코드: 지시문 48토큰 채움, 그래프 하나)
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
S=outputs/piper_sub2_model/merged
log "[sub2 + 보조 28항목 다시] 쥔 블록 판정 고침 (일꾼 4개)"
rm -rf outputs/piper/claim_s2p $LOCK
WS=""
for k in 1 2 3 4; do ( queue s2p $S piper_sub2apr "$AS" ) & WS="$WS $!"; sleep 30; done
wait $WS
board piper_sub2apr "sub2 단계 지시(원래 지시문 살림) + 정렬·놓기·내려가기·다가가기 (지연 9)"
PIPER_BLOCKS=1 $PY fail_classify.py outputs/piper_sub2apr --md outputs/piper/fail_piper_sub2apr.md > /dev/null
log "끝"
