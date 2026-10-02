#!/usr/bin/env bash
# 10/2 21시 — v6c 학습이 끝난 뒤 단계들을 **학습에 안 쓴 새 배치(장면 1000번대)** 로 평가한다.
#   이유: LIBERO 고정 장면은 번호 % 50 이라, 학습 데이터(장면 50~139)에 평가 장면 20~49 배치가 들어 있었다.
cd "$(dirname "$0")"
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
gpu_n(){ ps -eo comm,args | awk '$1=="python" && /switch_experiment\.py|dagger_v6\.py|a2c2\.py|train_lora\.py/' | wc -l; }
forget(){ local pol=$1 out=$2 ev=$3; shift 3; for t in "$@"; do $PY switch_experiment.py --policy $pol --task-a $t --strategy none $ev --out $out 2>&1 | grep -E "$F"; done; }
switch(){ local pol=$1 out=$2 ev=$3 st=$4; shift 4; for p in "$@"; do $PY switch_experiment.py --policy $pol --task-a ${p%:*} --task-b ${p#*:} --switch-at grasp:3 --strategy $st $ev --out $out 2>&1 | grep -E "$F"; done; }
P1="8:0 8:3 8:5 8:7 8:9 4:5"; P2="4:9 1:7 8:1 8:4 1:8 4:1"
while ps -eo comm,args | awk '$1=="python" && /train_lora\.py/' | grep -q .; do sleep 60; done
POL=outputs/v6c_model/merged
[ -f $POL/model.safetensors ] || { log "v6c 모델이 없다 — 멈춤"; exit 1; }
log "v6c 학습 끝"

log "[2] A2C2 데이터(v6c) + 빠른 비교 (새 배치 1000~1009, 계속 계산 n_act 1 vs 10)"
( $PY a2c2.py gen --policy $POL --ttrtc --data data/v6_normal data/v6_switch data/v6_dagger_lat --frac 0.5 --stride 16 \
    --batch 8 --out data/a2c2_v6c 2>&1 | grep -E "끝|Traceback|Error" ) &
EVQ="--episodes 10 --start-episode 1000 --latency-steps 11 --ttrtc"
( for NA in 1 10; do forget $POL outputs/v6ch_na$NA "$EVQ --n-action-steps $NA" 0 3 8; done ) &
( for NA in 1 10; do switch $POL outputs/v6ch_na$NA "$EVQ --n-action-steps $NA" keep 8:3 8:7; done ) &
wait
NA=$($PY - <<'PY'
import glob, json
def rate(d):
    k = n = 0
    for f in glob.glob(f"outputs/v6ch_na{d}/*.json"):
        for e in json.load(open(f))["episodes"]:
            key = "b_success" if e.get("switched") and e.get("task_b") else "a_success"
            k += bool(e.get(key)); n += 1
    return k / max(n, 1)
print(1 if rate(1) >= rate(10) else 10)
PY
)
log "빠른 비교 끝 → n_act $NA 로 정식 평가"

log "[3] A2C2 학습 + v6c 정식 평가 + v6b 같은 새 배치 평가 (비교 기준)"
EV="--episodes 10 --start-episode 1000 --latency-steps 11 --ttrtc --n-action-steps $NA"
EVB="--episodes 10 --start-episode 1000 --latency-steps 11"
( $PY a2c2.py train --gen data/a2c2_v6c --out outputs/a2c2_v6c --steps 30000 --batch 64 --workers 4 2>&1 \
    | grep --line-buffered -E '보정 네트워크|"step": [0-9]*0000,|저장|Traceback|Error' ) &
( forget $POL outputs/v6ch_forget "$EV" 0 1 2 3 4 5 6 7 8 9; switch $POL outputs/v6ch_switch "$EV" keep $P2 ) &
( switch $POL outputs/v6ch_switch "$EV" keep $P1 ) &
( forget outputs/v6b_model/merged outputs/v6bh_forget "$EVB" 0 1 2 3 4 5 6 7 8 9
  switch outputs/v6b_model/merged outputs/v6bh_switch "$EVB" flush $P1 $P2 ) &
wait
$PY scoreboard.py --forget outputs/v6ch_forget --switch outputs/v6ch_switch --name "v6c 새 배치 지연11" \
  --png outputs/media/score_v6ch.png --md outputs/v6/score_v6ch.md | tail -1

log "[4] v6c + A2C2 (그리퍼 보정 포함/제외)"
[ -f outputs/a2c2_v6c/head.pt ] || { log "보정 네트워크가 없다 — 멈춤"; exit 1; }
EVA="$EV --a2c2 outputs/a2c2_v6c"
( forget $POL outputs/v6cah_forget "$EVA" 0 1 2 3 4 5 6 7 8 9 ) &
( switch $POL outputs/v6cah_switch "$EVA" keep $P1 ) &
( switch $POL outputs/v6cah_switch "$EVA" keep $P2 ) &
( switch $POL outputs/v6cah_posrot "$EVA --a2c2-dims posrot" keep 8:7 1:7; forget $POL outputs/v6cah_posrot "$EVA --a2c2-dims posrot" 7 8 1 ) &
wait
$PY scoreboard.py --forget outputs/v6cah_forget --switch outputs/v6cah_switch --name "v6c + A2C2 새 배치 지연11" \
  --png outputs/media/score_v6cah.png --md outputs/v6/score_v6cah.md | tail -1
$PY scoreboard.py --forget outputs/v6bh_forget --switch outputs/v6bh_switch --name "v6b 새 배치 지연11" \
  --png outputs/media/score_v6bh.png --md outputs/v6/score_v6bh.md | tail -1
log "끝"
