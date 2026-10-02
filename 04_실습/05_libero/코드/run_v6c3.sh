#!/usr/bin/env bash
# 10/3 아침 — v6c2 도 쉬운 점검(T8 학습 배치·지연 없음) 0/5. 그릇 테두리보다 2~3cm 높은 곳에서 그리퍼를 닫아 헛잡는다.
# v6c·v6c2 공통점은 동작 정규화 다시 재기(--renorm-action). v6b(정규화 그대로)는 3/5.
# v6b 에서 시작하면서 정규화를 바꾸면 이미 배운 동작 출력의 기준이 통째로 바뀐다 → 정규화는 그대로 두고 나머지는 v6c2 와 같게.
cd "$(dirname "$0")"
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
DATA="data/v6_normal data/v6_switch data/v6_dagger_lat data/v7_normal"
log "[1] v6c3 학습 (v6b 에서, v6c2 와 같되 정규화 다시 재기 없음)"
$PY train_lora.py --policy outputs/v6b_model/merged --data $DATA \
  --full-expert --aug --balance --dagger-frac 0.3 --workers 0 --rtc-max-delay 14 --ema 0.999 --frame-stride 2 \
  --steps 24000 --batch-size 4 --grad-accum 2 --lr 5e-5 --eval-every 3000 --save-every 6000 --log-every 500 \
  --out outputs/v6c3_model 2>&1 | grep --line-buffered -E '지연 흉내|정규화 다시|교정 시범 비율|에피소드 학습|val_loss|모델 저장|Traceback|Error'
[ -f outputs/v6c3_model/merged/model.safetensors ] || { log "v6c3 모델이 없다 — 멈춤"; exit 1; }
log "[2] 새 배치 평가 + A2C2"
./run_post.sh v6c3 $DATA > outputs/v6/run_post_v6c3.log 2>&1
grep -qE "^=== [0-9/]+ [0-9:]+ 끝$" outputs/v6/run_post_v6c3.log || { log "평가 단계 멈춤 — run_post_v6c3.log 확인"; exit 1; }
log "[3] 라운드 v6d, v6e"
./run_round.sh v6c3 v6d v6c3h v6c3ah outputs/a2c2_v6c3 > outputs/v6/round_v6d.log 2>&1
grep -q "라운드 끝" outputs/v6/round_v6d.log || exit 1
./run_round.sh v6d v6e v6dh v6dah outputs/a2c2_v6d > outputs/v6/round_v6e.log 2>&1
log "끝"
