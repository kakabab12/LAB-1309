#!/usr/bin/env bash
# 내려가기 재촉(⑦)·다가가기(⑧) 보조 시험 (10/17): als2 + 정렬·놓기 보조 28항목(run_alc3_probe.sh)이 끝나면,
#   같은 장면(1000~1019)에서 als2 + 정렬·놓기 + ⑦  /  + ⑦ + ⑧ 을 6항목 (T0·T8, 일 바꾸기·돌아오기 8:5·4:9·8:9·4:5)
#   비교 기준은 28항목 평가의 als2place 같은 항목
# run_apr28.sh (10/17 05:40): 위 6항목에서 + 내려가기·다가가기가 일 바꾸기 4항목 B 20 → 37/80, 돌아오기 13 → 28/80.
#   → 28항목 전체. run_alc3_probe.sh(1638640)·run_assist2_test.sh(1687720) 가 끝나면. 이미 잰 항목은 건너뜀
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

for p in 1638640 1687720; do while kill -0 $p 2>/dev/null; do sleep 60; done; done
rm -rf outputs/piper/claim_apr28 $LOCK; mkdir -p outputs/piper/claim_apr28
for f in outputs/piper_als2apr_forget/A*_Bnone_*.json; do [ -f "$f" ] || continue; t=$(basename $f | sed -E 's/^A([0-9])_.*/\1/'); mkdir -p outputs/piper/claim_apr28/$t; done
for f in outputs/piper_als2apr_switch/A*_B*_*.json; do [ -f "$f" ] || continue; p=$(basename $f | sed -E 's/^A([0-9])_B([0-9])_.*/\1:\2/'); mkdir -p "outputs/piper/claim_apr28/$p"; done
log "이미 잰 항목: $(ls outputs/piper/claim_apr28 | tr '\n' ' ')"
A2=outputs/piper_als2_model/merged
log "[als2 + 정렬·놓기·내려가기·다가가기 28항목] (일꾼 4개)"
WS=""
for k in 1 2 3 4; do ( queue apr28 $A2 piper_als2apr "--approach-assist --descend-assist" ) & WS="$WS $!"; sleep 30; done
wait $WS
board piper_als2apr "als2 + 정렬·놓기·내려가기·다가가기 보조 (지연 9)"
for o in piper_als2h piper_als2place piper_als2apr; do PIPER_BLOCKS=1 $PY fail_classify.py outputs/$o --md outputs/piper/fail_$o.md > /dev/null; done
log "끝"
