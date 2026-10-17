#!/usr/bin/env bash
# 보조를 켠 채 교정 시범 (10/17 12시): als2 + 정렬·놓기·내려가기·다가가기 (28항목 61.4%) 의 남은 실패 대부분이
#   '블록 위에서 못 쥠' — 쥐는 높이까지 내려와도 학생이 오므리지 않고 손을 올린다. 쥐기를 규칙으로 해 줘도(⑩) 그 뒤 들어 올리지 않는다.
#   → 학생이 보조와 함께 실제로 가는 상태에서 시범 프로그램이 넘겨받은 동작을 배우게 한다 (배치 조건 = 보조 켬).
#   ① 손목 맞추기 시험(run_yaw_test.sh 2079248) 끝 → ② als2 교정 시범 (보조 켬, 항목은 als2apr 성적으로) → ③ 정리
#   ④ als3a 학습 (als2 에서) → ⑤ 민감도 → ⑥ als3a + 보조 28항목 → als3a 단독 28항목
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


while kill -0 2079248 2>/dev/null; do sleep 60; done
B=outputs/piper_als2_model/merged; OUTD=data/piper_al_dagger_als2a
eval "$($PY piper_sim/plan_dagger.py --base piper_als2apr --start 4100)"
log "[als2 교정 시범, 보조 켬] 단독 30:[${T30}] 20:[${T20}] 10:[${T10}] / 전환 30:[${P30}] 20:[${P20}] 10:[${P10}] / 재개 30:[${R30}] 20:[${R20}]"
DG="--policy $B $DGQ $AS --strategy keep --max-policy 350 --out $OUTD"
dag_t(){ for k in 30 20 10; do v=T$k; e=E$k; [ -n "${!v}" ] && $PY piper_sim/dagger_piper.py $DG --tasks ${!v} --episodes ${!e} --seed $((301 + k)); done; }
dag_p(){ for k in 30 20 10; do v=P$k; e=E$k; [ -n "${!v}" ] && $PY piper_sim/dagger_piper.py $DG --pairs ${!v} --episodes ${!e} --seed $((302 + k)); done; }
dag_r(){ for k in 30 20 10; do v=R$k; e=E$k; [ -n "${!v}" ] && $PY piper_sim/dagger_piper.py $DG --pairs ${!v} --resume --episodes ${!e} --seed $((303 + k)); done; }
( dag_t > outputs/piper/dagger_als2a_t.log 2>&1 ) & D1=$!
( dag_p > outputs/piper/dagger_als2a_p.log 2>&1 ) & D2=$!
( dag_r > outputs/piper/dagger_als2a_r.log 2>&1 ) & D3=$!
wait $D1 $D2 $D3
log "als2 교정 시범 끝: $(ls $OUTD/episodes | wc -l)"
prep_data $OUTD $B
DATA="$DATA ${OUTD}_t"
log "[als3a 학습] als2 에서 (+ 보조 켠 교정 시범)"
train $B outputs/piper_als3a_model 10000 3e-5 0.35
C=outputs/piper_als3a_model/merged
[ -f $C/sufstate ] || { log "als3a 모델 없음 — 멈춤"; exit 1; }
log "[als3a + 보조 28항목] → [als3a 단독 28항목] (일꾼 4개)"
rm -rf outputs/piper/claim_a3p outputs/piper/claim_a3h $LOCK
worker(){ queue a3p $C piper_als3aapr "$AS"; queue a3h $C piper_als3ah ""; }
WS=""
for k in 1 2 3 4; do ( worker ) & WS="$WS $!"; sleep 30; done
wait $WS
board piper_als3aapr "als3a + 정렬·놓기·내려가기·다가가기 (지연 9)"
board piper_als3ah "als3a 단독 (지연 9)"
for o in piper_als3aapr piper_als3ah; do PIPER_BLOCKS=1 $PY fail_classify.py outputs/$o --md outputs/piper/fail_$o.md > /dev/null; done
log "끝"
