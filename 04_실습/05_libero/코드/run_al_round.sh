#!/usr/bin/env bash
# 새 쥐기 방식(align) 라운드 (10/14)
#   ① align 시범 수집이 끝나면 (run_align_collect.sh) 사진 특징 미리 계산 (vis_cache, 학습 약 3배 빠름·사진을 메모리에 안 올림)
#   ② al1: bo1 에서 이어 align 시범 + bo1 교정 시범(align 으로 넘겨받음)으로 학습 (쥐는 방식이 두 갈래로 섞이지 않게 — ACT 논문 '늘 같은 방식')
#   ③ al1 28항목 (CUDA 그래프, 실제 지연 452ms = 9스텝)
set -u
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
ngpu(){ nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -vc rustdesk; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --log-every 500"
AL="data/piper_al_normal_t data/piper_al_wide_normal_t data/piper_al_switch_t data/piper_al_recovery_t data/piper_al_dagger_bo1_t"
ev_t(){ local pol=$1 o=$2; shift 2; for t in "$@"; do $PY piper_sim/eval_piper.py --policy $pol --task-a $t --strategy none --episodes 20 --start-episode 1000 $EVQ --out outputs/${o}_forget 2>&1 | grep -E "$F"; done; }
ev_p(){ local pol=$1 o=$2; shift 2; for p in "$@"; do $PY piper_sim/eval_piper.py --policy $pol --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy keep --episodes 20 --start-episode 1000 $EVQ --out outputs/${o}_switch 2>&1 | grep -E "$F"; done; }
board(){ $PY scoreboard.py --blocks --forget outputs/$1_forget --switch outputs/$1_switch --name "$2" --png outputs/media/score_$1.png --md outputs/piper/score_$1.md | tail -1
         $PY scoreboard.py --blocks --loose --forget outputs/$1_forget --switch outputs/$1_switch --name "$2 판 안 기준" --png outputs/media/score_$1_loose.png --md outputs/piper/score_$1_loose.md | tail -1; }
all28(){ local pol=$1 o=$2
  until [ "$(ngpu)" -le 0 ]; do sleep 60; done
  ( ev_t $pol $o 0 1 2 3 4 ) & local P1=$!; ( ev_t $pol $o 5 6 7 8 9 ) & local P2=$!
  ( ev_p $pol $o 8:0 8:3 8:5 8:7 8:9 4:5 ) & local P3=$!; ( ev_p $pol $o 4:9 1:7 8:1 8:4 1:8 4:1 ) & local P4=$!
  wait $P1 $P2 $P3 $P4; }

until grep -q "al 검출 미리 계산 끝" outputs/piper/al_collect.log 2>/dev/null; do sleep 120; done
until grep -q "교정 시범 끝" outputs/piper/al_dagger.log 2>/dev/null; do sleep 120; done
log "[1] 사진 특징 미리 계산 (align 시범 $(ls data/piper_al_*_t/episodes/*.npz | wc -l)개)"
until [ "$(ngpu)" -le 2 ]; do sleep 60; done
( $PY vis_cache.py build --policy outputs/piper_bo1_model/merged --data $AL --out cache/vis --shard 0 --nshard 2 2>&1 | grep -E "끝|Error|Traceback" ) & Q1=$!
( $PY vis_cache.py build --policy outputs/piper_bo1_model/merged --data $AL --out cache/vis --shard 1 --nshard 2 2>&1 | grep -E "끝|Error|Traceback" ) & Q2=$!
wait $Q1 $Q2
log "[2] al1 학습 (bo1 에서, align 시범만, 사진 특징 미리 계산본, 모든 프레임)"
$PY train_lora.py --policy outputs/piper_bo1_model/merged --data $AL $TR --abs-pos --objfeat --vis-cache cache/vis --frame-stride 1 \
  --dagger-frac 0.25 --batch-size 8 --grad-accum 2 --steps 15000 --lr 5e-5 --eval-every 2500 --save-every 5000 \
  --out outputs/piper_al1_model 2>&1 | grep --line-buffered -E '에피소드 학습|물체 위치|사진 특징|val_loss|모델 저장|Traceback|Error'
B=outputs/piper_al1_model/merged
[ -f $B/objfeat ] || { log "al1 모델 없음 — 멈춤"; exit 1; }
log "[3] al1 28항목 (지연 9, CUDA 그래프)"
all28 $B piper_al1h
board piper_al1h "align 시범 al1 (새 배치 20장면, 지연 9)"
log "끝"
