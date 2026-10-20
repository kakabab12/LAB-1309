#!/usr/bin/env bash
# 받침대 없음 1라운드 (10/20 저녁, 사용자 "앞으로 받침대 없이"): PIPER_NOPED=1 — 밑동 (-0.33, 0, 책상 높이), 처음 자세 책상 위 18cm·앞으로 45°,
#   쥐는 방향은 '똑바로 아래' 기준 ±35°, 막대는 2.8cm 에서 쥠. 시범 프로그램 시험: 과제 10개 모두, 일 바꾸기·돌아오기도 저장됨.
#   ① 시범 전부 새로 (정상 600·넓은 600·일 바꾸기 12쌍x40·회복 300 — 회복은 PIPER_NEAR_GRASP=1) → 다듬기·검출·손 기준 위치
#   ② 올라가는 시범 거르기 → ③ GPU 가 비면(받침대 als3b 평가 1044443 끝) 사진 특징 → ④ np1 학습 (als2 에서, 동작 정규화 다시)
#   ⑤ np1 + 보조 28항목 → np1 단독 28항목
set -u
# run_np_round_c.sh (10/20 22:45): 일 바꾸기도 미리 계산 (np_pre2.log) → 회복만 다듬음
# (이전) run_np_round_b.sh (10/20 21:45): 본체만 바꿈 — 수집기(1063228 일 바꾸기→회복, 1142252 일 바꾸기 뒤쪽)와 정상·넓은 배치 미리 계산(1313996)을 기다리고,
#   정상·넓은 배치는 다시 다듬지 않는다 (미리 계산과 겹쳐 폴더를 지우지 않게)
cd "$(dirname "$0")"
export PIPER_NOPED=1 PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl
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



for p in 1063228 1142252 1313996 1347059; do while kill -0 $p 2>/dev/null; do sleep 60; done; done
log "수집 끝: 정상 $(ls data/piper_np_normal/episodes | wc -l), 넓은 $(ls data/piper_np_wide_normal/episodes | wc -l), 전환·재개 $(ls data/piper_np_switch/episodes | wc -l), 회복 $(ls data/piper_np_recovery/episodes | wc -l)"
ND=""
for d in recovery; do rm -rf data/piper_np_${d}_t; $PY piper_sim/trim_stalls.py data/piper_np_$d data/piper_np_${d}_t | tail -1; ND="$ND data/piper_np_${d}_t"; done
for s in 0 1 2; do ( $PY objfeat_cache.py --data $ND --shard $s --nshard 3 > /dev/null 2>&1 ) & done; wait
$PY relfeat.py --data $ND
NPD="data/piper_np_normal_t data/piper_np_wide_normal_t data/piper_np_switch_t $ND"
log "[올라가는 시범 거르기]"
touch outputs/piper/exclude_rise_np.txt
$PY find_rise.py $NPD --append outputs/piper/exclude_rise_np.txt
log "[사진 특징] (GPU, 미리 계산한 것은 건너뜀)"
B=outputs/piper_als2_model/merged
( $PY vis_cache.py build --policy $B --data $NPD --out cache/vis --batch 8 --stride 3 --shard 0 --nshard 2 2>&1 | grep -E "끝|Error|Traceback" ) & Q1=$!
( $PY vis_cache.py build --policy $B --data $NPD --out cache/vis --batch 8 --stride 3 --shard 1 --nshard 2 2>&1 | grep -E "끝|Error|Traceback" ) & Q2=$!
wait $Q1 $Q2
DATA="$NPD"
log "[np1 학습] als2 에서, 받침대 없음 시범만, 동작 정규화 다시, 15000스텝"
train $B outputs/piper_np1_model 15000 3e-5 0.35 --exclude outputs/piper/exclude_rise_np.txt --renorm-action
C=outputs/piper_np1_model/merged
[ -f $C/sufstate ] && [ -f $C/relfeat ] || { log "np1 모델 없음 또는 relfeat 표시 없음 — 멈춤"; exit 1; }
TASKS="0 1 2 3 4 5 6 7 8 9"; PAIRS="8:0 8:3 8:5 8:7 8:9 4:5 4:9 1:7 8:1 8:4 1:8 4:1"
log "[np1 + 보조 28항목] → [np1 단독 28항목] (일꾼 4개)"
rm -rf outputs/piper/claim_n1p outputs/piper/claim_n1h $LOCK
worker(){ queue n1p $C piper_np1apr "$AS"; queue n1h $C piper_np1h ""; }
WS=""
for k in 1 2 3 4; do ( worker ) & WS="$WS $!"; sleep 30; done
wait $WS
board piper_np1apr "받침대 없음 np1 + 정렬·놓기·내려가기·다가가기 (지연 9)"
board piper_np1h "받침대 없음 np1 단독 (지연 9)"
for o in piper_np1apr piper_np1h; do PIPER_BLOCKS=1 $PY fail_classify.py outputs/$o --md outputs/piper/fail_$o.md > /dev/null; done
log "끝"
