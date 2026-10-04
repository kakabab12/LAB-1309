#!/usr/bin/env bash
# 블록 라운드 C (10/4 22:40): b1 이 0~20% 인 원인을 고친 판.
#   ① 평가 버그 고침: 지연 흉내에서 시작할 때 첫 계획의 앞 11개(블록 쪽 옆 이동)를 버리던 것 (switch_experiment.policy_action_async)
#   ② 사진 위치 흔들기(±8픽셀 ≈ 책상 위 3cm) 끄기 — 블록 위치 정밀도를 흐린다는 가설
#   ③ 학습량: 블록 시범 2배(1·2차 수집) + 더 길게, b1 에서 이어서 → b1c
#   ④ b1c 학습과 나란히 b1 로 교정 시범(DAgger) — 빗나감·빈손 다시 집기를 가르친다
#   ⑤ b2 = b1c + 교정 시범 → 새 배치 28항목 → A2C2 → 다시 평가
set -u
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 MUJOCO_GL=egl
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
B1=outputs/piper_b1_model/merged
EVQ="--latency-steps 11 --ttrtc --n-action-steps 1"
DATA="data/piper_blk_normal_t data/piper_blk_switch_t data/piper_blk_normal2_t data/piper_blk_switch2_t"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --frame-stride 2 --batch-size 4 --grad-accum 2 --log-every 500"
log "[1] b1c 학습 (b1 에서, 사진 흔들기 끔, 블록 시범 2배, 24000스텝) ‖ b1 교정 시범"
touch outputs/piper/TRAINING
( $PY train_lora.py --policy $B1 --data $DATA $TR --steps 24000 --lr 5e-5 --eval-every 3000 --save-every 4000 \
    --out outputs/piper_b1c_model 2>&1 | grep --line-buffered -E '에피소드 학습|val_loss|모델 저장|Traceback|Error' > outputs/piper/train_b1c.log ) & PT=$!
OUTD=data/piper_blk_dagger_b1
D="--policy $B1 $EVQ --strategy keep --out $OUTD"
( $PY piper_sim/dagger_piper.py $D --tasks 0 1 2 3 4 5 6 7 8 9 --episodes 2200-2229 --seed 41 > outputs/piper/dagger_b1_t.log 2>&1 ) & P1=$!
( $PY piper_sim/dagger_piper.py $D --pairs 8:0,8:3,8:5,8:7,8:9,4:5,4:9,1:7,8:1,8:4,1:8,4:1 --episodes 2200-2219 --seed 42 > outputs/piper/dagger_b1_p.log 2>&1 ) & P2=$!
( $PY piper_sim/dagger_piper.py $D --pairs 8:5,8:7,8:9,4:5,4:9,1:7 --resume --episodes 2230-2249 --seed 43 > outputs/piper/dagger_b1_r.log 2>&1 ) & P3=$!
wait $P1 $P2 $P3
log "교정 시범 끝: $(ls $OUTD/episodes 2>/dev/null | wc -l) — $(grep -h STATS outputs/piper/dagger_b1_*.log | tr '\n' ' ')"
$PY piper_sim/trim_stalls.py $OUTD ${OUTD}_t
# 비교 기준: b1 을 고친 평가 방식으로, 몇 항목만 (학습이 도는 동안 GPU 남는 자리)
ev_t(){ local pol=$1 o=$2 opt=$3; shift 3; for t in "$@"; do $PY piper_sim/eval_piper.py --policy $pol --task-a $t --strategy none --episodes 20 --start-episode 1000 $opt --out outputs/${o}_forget 2>&1 | grep -E "$F"; done; }
ev_p(){ local pol=$1 o=$2 opt=$3; shift 3; for p in "$@"; do $PY piper_sim/eval_piper.py --policy $pol --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy keep --episodes 20 --start-episode 1000 $opt --out outputs/${o}_switch 2>&1 | grep -E "$F"; done; }
log "[2] b1 (고친 평가) 일부 항목"
( ev_t $B1 piper_b1fx "$EVQ" 1 4 8 9 ) & P1=$!; ( ev_p $B1 piper_b1fx "$EVQ" 8:1 8:5 4:9 ) & P2=$!
wait $P1 $P2
wait $PT
rm -f outputs/piper/TRAINING
B1C=outputs/piper_b1c_model/merged
[ -f $B1C/model.safetensors ] || { log "b1c 모델 없음 — 멈춤"; exit 1; }
log "[3] b1c 같은 항목 ‖ b2 학습 (b1c 에서 + 교정 시범)"
touch outputs/piper/TRAINING
( $PY train_lora.py --policy $B1C --data $DATA ${OUTD}_t $TR --dagger-frac 0.3 --steps 16000 --lr 3e-5 --eval-every 2000 --save-every 4000 \
    --out outputs/piper_b2_model 2>&1 | grep --line-buffered -E '에피소드 학습|val_loss|모델 저장|Traceback|Error' > outputs/piper/train_b2.log ) & PT=$!
( ev_t $B1C piper_b1cx "$EVQ" 1 4 8 9 ) & P1=$!; ( ev_p $B1C piper_b1cx "$EVQ" 8:1 8:5 4:9 ) & P2=$!
wait $P1 $P2
$PY piper_sim/mean_score.py piper_b1fx piper_b1cx | while read l; do log "  평균 $l"; done
# b1c 로 A2C2 데이터 (b2 학습 동안)
$PY a2c2.py gen --policy $B1C --ttrtc --data $DATA --frac 0.5 --stride 16 --batch 8 --out data/a2c2_piper_b1c 2>&1 | grep -E "끝|Traceback|Error"
wait $PT
rm -f outputs/piper/TRAINING
POL=outputs/piper_b2_model/merged
[ -f $POL/model.safetensors ] || { log "b2 모델 없음 — 멈춤"; exit 1; }
log "[4] b2 새 배치 28항목 ‖ b2 A2C2 데이터"
( ev_t $POL piper_b2h "$EVQ" 0 1 2 3 4 5 6 7 8 9 ) & P1=$!; ( ev_p $POL piper_b2h "$EVQ" 8:0 8:3 8:5 8:7 8:9 4:5 ) & P2=$!; ( ev_p $POL piper_b2h "$EVQ" 4:9 1:7 8:1 8:4 1:8 4:1 ) & P3=$!
$PY a2c2.py gen --policy $POL --ttrtc --data $DATA ${OUTD}_t --frac 0.5 --stride 16 --batch 8 --out data/a2c2_piper_b2 2>&1 | grep -E "끝|Traceback|Error"
$PY a2c2.py train --gen data/a2c2_piper_b2 --out outputs/a2c2_piper_b2 --steps 30000 --batch 64 --workers 4 2>&1 | grep -E "끝|val|Traceback|Error" | tail -3
wait $P1 $P2 $P3
$PY scoreboard.py --forget outputs/piper_b2h_forget --switch outputs/piper_b2h_switch --name "블록 2차 piper_b2 (새 배치 20장면, 지연 11)" \
  --png outputs/media/score_piper_b2.png --md outputs/piper/score_piper_b2.md | tail -1
log "[5] b2 + A2C2"
A="--a2c2 outputs/a2c2_piper_b2"
( ev_t $POL piper_b2ah "$EVQ $A" 0 1 2 3 4 5 6 7 8 9 ) & P1=$!; ( ev_p $POL piper_b2ah "$EVQ $A" 8:0 8:3 8:5 8:7 8:9 4:5 ) & P2=$!; ( ev_p $POL piper_b2ah "$EVQ $A" 4:9 1:7 8:1 8:4 1:8 4:1 ) & P3=$!
wait $P1 $P2 $P3
$PY scoreboard.py --forget outputs/piper_b2ah_forget --switch outputs/piper_b2ah_switch --name "블록 2차 + A2C2 (새 배치 20장면, 지연 11)" \
  --png outputs/media/score_piper_b2a.png --md outputs/piper/score_piper_b2a.md | tail -1
log "끝"
