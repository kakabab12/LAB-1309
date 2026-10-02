#!/usr/bin/env bash
# 10/2 — v6c: 학습 때 지연 흉내내기(training-time RTC) + EMA + 지연 조건 DAgger 데이터, 그다음 A2C2 보정 네트워크
# DAgger 수집(run_dagger_v6b.sh)이 끝나면 시작한다. VLA 학습은 혼자 돈다 (메모리 18GB). GPU 프로세스는 3개까지.
cd "$(dirname "$0")"
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
busy(){ ps -eo comm,args | awk '$1=="python" && /dagger_v6\.py|switch_experiment\.py|collect_v6\.py|train_lora\.py/ && !/awk/' | grep -q .; }
forget(){ local pol=$1 out=$2 ev=$3; shift 3; for t in "$@"; do $PY switch_experiment.py --policy $pol --task-a $t --strategy none $ev --out $out 2>&1 | grep -E "$F"; done; }
switch(){ local pol=$1 out=$2 ev=$3 st=$4; shift 4; for p in "$@"; do $PY switch_experiment.py --policy $pol --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy $st $ev --out $out 2>&1 | grep -E "$F"; done; }
PAIRS="8:0 8:3 8:5 8:7 8:9 4:5 4:9 1:7 8:1 8:4 1:8 4:1"

while busy; do sleep 60; done
log "DAgger 끝: $(ls data/v6_dagger_lat/episodes | wc -l) 시범"

log "[1] v6c 학습 (v6b에서 이어서 24000스텝)"
$PY train_lora.py --policy outputs/v6b_model/merged --data data/v6_normal data/v6_switch data/v6_dagger_lat \
  --full-expert --aug --balance --workers 0 --rtc-max-delay 14 --ema 0.999 \
  --steps 24000 --batch-size 4 --grad-accum 2 --lr 3e-5 --eval-every 3000 --save-every 6000 --log-every 500 \
  --out outputs/v6c_model 2>&1 | grep --line-buffered -E '지연 흉내|에피소드 학습|val_loss|모델 저장|Traceback|Error'
POL=outputs/v6c_model/merged
[ -f $POL/model.safetensors ] || { log "v6c 모델이 없다 — 멈춤"; exit 1; }

log "[2] A2C2 데이터 만들기 + 지연 평가 빠른 비교 (계속 계산 n_act 1 vs 10스텝마다)"
( $PY a2c2.py gen --policy $POL --ttrtc --data data/v6_normal data/v6_switch data/v6_dagger_lat --frac 0.5 --stride 16 \
    --batch 8 --out data/a2c2_v6c 2>&1 | grep -E "끝|Traceback|Error" ) &
EVQ="--episodes 10 --start-episode 20 --latency-steps 11 --ttrtc"
( for NA in 1 10; do forget $POL outputs/v6clat_na$NA "$EVQ --n-action-steps $NA" 0 3 8; done ) &
( for NA in 1 10; do switch $POL outputs/v6clat_na$NA "$EVQ --n-action-steps $NA" keep 8:3 8:7; done ) &
wait
NA=$($PY - <<'PY'
import glob, json
def rate(d):
    k = n = 0
    for f in glob.glob(f"outputs/v6clat_na{d}/*.json"):
        for e in json.load(open(f))["episodes"]:
            key = "b_success" if e.get("switched") else "a_success"
            k += bool(e.get(key)); n += 1
    return k / max(n, 1)
r1, r10 = rate(1), rate(10)
print(1 if r1 >= r10 else 10)
PY
)
log "빠른 비교 끝 → n_act $NA 로 정식 평가"

log "[3] A2C2 학습 + v6c 정식 지연 평가 (장면 20~29, 10회, 전환은 멈춤 없는 keep)"
EV="--episodes 10 --start-episode 20 --latency-steps 11 --ttrtc --n-action-steps $NA"
( $PY a2c2.py train --gen data/a2c2_v6c --out outputs/a2c2_v6c --steps 30000 --batch 64 --workers 4 2>&1 \
    | grep --line-buffered -E '보정 네트워크|"step": [0-9]*000,|저장|Traceback|Error' ) &
( forget $POL outputs/v6clat_forget "$EV" 0 1 2 3 4 5 6 7 8 9 ) &
( switch $POL outputs/v6clat_switch "$EV" keep $PAIRS ) &
wait
$PY scoreboard.py --forget outputs/v6clat_forget --switch outputs/v6clat_switch --name "v6c 지연11" \
  --png outputs/media/score_v6c_lat.png --md outputs/v6/score_v6c_lat.md | tail -2

log "[4] v6c + A2C2 지연 평가"
[ -f outputs/a2c2_v6c/head.pt ] || { log "보정 네트워크가 없다 — 멈춤"; exit 1; }
EVA="$EV --a2c2 outputs/a2c2_v6c"
( forget $POL outputs/v6ca_forget "$EVA" 0 1 2 3 4 5 6 7 8 9 ) &
( switch $POL outputs/v6ca_switch "$EVA" keep 8:0 8:3 8:5 8:7 8:9 4:5 ) &
( switch $POL outputs/v6ca_switch "$EVA" keep 4:9 1:7 8:1 8:4 1:8 4:1 ) &
wait
$PY scoreboard.py --forget outputs/v6ca_forget --switch outputs/v6ca_switch --name "v6c + A2C2 지연11" \
  --png outputs/media/score_v6ca_lat.png --md outputs/v6/score_v6ca_lat.md | tail -2
log "끝"
