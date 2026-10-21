#!/usr/bin/env bash
# 받침대 없음 2라운드 (10/21 08시): np1 + 보조 71.8% (혼자 92.0, 일 바꾸기 73.3, 돌아오기 35.0). 돌아오기와 약한 쌍(새 지시가 노란 막대) 올리기.
#   ① np1 단독 평가(run_np_round_d.sh 1442004) 끝 → ② np1 교정 시범 (보조 켬, 고친 시범 프로그램 PIPER_NEAR_GRASP=1, np1apr 성적으로)
#      ‖ 돌아오기 쌍 시범 더 (장면 3140~3179, 6쌍) → ③ 정리·올라가는 시범 거르기 → ④ np2 학습 (np1 에서, 10,000스텝)
#   ⑤ np2 + 보조 28항목 → np2 단독 28항목
set -u
# run_np_round_d.sh (10/21 01:45): 사진 특징 계산을 여러 줄이 같은 파일에 동시에 쓰다 한 줄이 죽었다 (tmp 이름 바꾸기 충돌) →
#   남은 두 줄(1410099, 1413204)이 끝나면 한 줄로 다시 훑어 빠진 것을 채운 뒤 학습
# (이전) run_np_round_c.sh (10/20 22:45): 일 바꾸기도 미리 계산 (np_pre2.log) → 회복만 다듬음
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



NPD="data/piper_np_normal_t data/piper_np_wide_normal_t data/piper_np_switch_t data/piper_np_recovery_t"
while kill -0 1442004 2>/dev/null; do sleep 60; done
B=outputs/piper_np1_model/merged; OUTD=data/piper_np_dagger_np1
eval "$($PY piper_sim/plan_dagger.py --base piper_np1apr --start 4500)"
log "[np1 교정 시범, 보조 켬, 시범 프로그램 고침] 단독 30:[${T30}] 20:[${T20}] 10:[${T10}] / 전환 30:[${P30}] 20:[${P20}] 10:[${P10}] / 재개 30:[${R30}] 20:[${R20}] 10:[${R10}]"
DG="--policy $B $DGQ $AS --strategy keep --max-policy 350 --out $OUTD"
dag_t(){ for k in 30 20 10; do v=T$k; e=E$k; [ -n "${!v}" ] && PIPER_NEAR_GRASP=1 $PY piper_sim/dagger_piper.py $DG --tasks ${!v} --episodes ${!e} --seed $((501 + k)); done; }
dag_p(){ for k in 30 20 10; do v=P$k; e=E$k; [ -n "${!v}" ] && PIPER_NEAR_GRASP=1 $PY piper_sim/dagger_piper.py $DG --pairs ${!v} --episodes ${!e} --seed $((502 + k)); done; }
dag_r(){ for k in 30 20 10; do v=R$k; e=E$k; [ -n "${!v}" ] && PIPER_NEAR_GRASP=1 $PY piper_sim/dagger_piper.py $DG --pairs ${!v} --resume --episodes ${!e} --seed $((503 + k)); done; }
( dag_t > outputs/piper/dagger_np1_t.log 2>&1 ) & D1=$!
( dag_p > outputs/piper/dagger_np1_p.log 2>&1 ) & D2=$!
( dag_r > outputs/piper/dagger_np1_r.log 2>&1 ) & D3=$!
( $PY piper_sim/collect_piper.py switch --pairs 8:5,8:7,8:9,4:5,4:9,1:7 --episodes 3140-3179 --dart 0.1 --seed 216 --out data/piper_np_switch2 > outputs/piper/np_collect_s3.log 2>&1 ) & D4=$!
wait $D1 $D2 $D3 $D4
log "교정 시범 $(ls $OUTD/episodes | wc -l), 돌아오기 쌍 시범 더 $(ls data/piper_np_switch2/episodes | wc -l)"
prep_data $OUTD $B
prep_data data/piper_np_switch2 $B
$PY find_rise.py ${OUTD}_t data/piper_np_switch2_t --append outputs/piper/exclude_rise_np.txt
DATA="$NPD ${OUTD}_t data/piper_np_switch2_t"
log "[np2 학습] np1 에서, + 교정 시범 + 돌아오기 시범, 10000스텝"
train $B outputs/piper_np2_model 10000 3e-5 0.35 --exclude outputs/piper/exclude_rise_np.txt
C=outputs/piper_np2_model/merged
[ -f $C/sufstate ] && [ -f $C/relfeat ] || { log "np2 모델 없음 또는 relfeat 표시 없음 — 멈춤"; exit 1; }
TASKS="0 1 2 3 4 5 6 7 8 9"; PAIRS="8:0 8:3 8:5 8:7 8:9 4:5 4:9 1:7 8:1 8:4 1:8 4:1"
log "[np2 + 보조 28항목] → [np2 단독 28항목] (일꾼 4개)"
rm -rf outputs/piper/claim_n2p outputs/piper/claim_n2h $LOCK
worker(){ queue n2p $C piper_np2apr "$AS"; queue n2h $C piper_np2h ""; }
WS=""
for k in 1 2 3 4; do ( worker ) & WS="$WS $!"; sleep 30; done
wait $WS
board piper_np2apr "받침대 없음 np2 + 정렬·놓기·내려가기·다가가기 (지연 9)"
board piper_np2h "받침대 없음 np2 단독 (지연 9)"
for o in piper_np2apr piper_np2h; do PIPER_BLOCKS=1 $PY fail_classify.py outputs/$o --md outputs/piper/fail_$o.md > /dev/null; done
log "끝"
