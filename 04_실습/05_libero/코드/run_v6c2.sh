#!/usr/bin/env bash
# 10/3 새벽 — v6c 는 "쥔 물체를 바로 놓는" 교정 시범 267개 때문에 그릇을 집자마자 놓는 버릇이 생겼다.
# 그 시범을 격리하고(data/v6_dagger_lat_quarantine), 새 무작위 배치 시범(v7_normal)을 더해 v6b 에서 다시 학습한다.
cd "$(dirname "$0")"
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
until grep -q "전환·재개 시범" outputs/v6/v7_collect.log; do sleep 60; done     # 정상 수행 시범 수집이 끝날 때까지
log "v7 정상 시범 $(ls data/v7_normal/episodes | wc -l) 개"
DATA="data/v6_normal data/v6_switch data/v6_dagger_lat data/v7_normal"
log "[1] v6c2 학습 (v6b 에서, 격리 후 데이터 + 새 배치, 프레임 2개마다 1개)"
$PY train_lora.py --policy outputs/v6b_model/merged --data $DATA \
  --full-expert --aug --balance --dagger-frac 0.3 --workers 0 --rtc-max-delay 14 --ema 0.999 --renorm-action --frame-stride 2 \
  --steps 24000 --batch-size 4 --grad-accum 2 --lr 5e-5 --eval-every 3000 --save-every 6000 --log-every 500 \
  --out outputs/v6c2_model 2>&1 | grep --line-buffered -E '지연 흉내|정규화 다시|교정 시범 비율|에피소드 학습|val_loss|모델 저장|Traceback|Error'
[ -f outputs/v6c2_model/merged/model.safetensors ] || { log "v6c2 모델이 없다 — 멈춤"; exit 1; }
log "[2] 새 배치 평가 + A2C2"
./run_post.sh v6c2 $DATA > outputs/v6/run_post_v6c2.log 2>&1
grep -qE "^=== [0-9/]+ [0-9:]+ 끝$" outputs/v6/run_post_v6c2.log || { log "평가 단계 멈춤 — run_post_v6c2.log 확인"; exit 1; }
log "[3] 라운드 v6d, v6e"
./run_round.sh v6c2 v6d v6c2h v6c2ah outputs/a2c2_v6c2 > outputs/v6/round_v6d.log 2>&1
grep -q "라운드 끝" outputs/v6/round_v6d.log || exit 1
./run_round.sh v6d v6e v6dh v6dah outputs/a2c2_v6d > outputs/v6/round_v6e.log 2>&1
log "끝"
