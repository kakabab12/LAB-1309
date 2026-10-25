#!/usr/bin/env bash
# w1b (10/25 15시): w1 이 단독 0% (k=1 도 0/20) — 예전 찾기로 만든 wfeat 가 지시문 물체를 자주 아예 못 찾았다 (원기둥 31%).
#   + 물체 입력 열이 3e-5 로는 0.25 까지만 자람 (로봇 상태 열 3.3) → state_proj·state_suffix_proj 학습률 3e-4 (--proj-lr --new-lr).
#   고친 찾기의 wfeat 로 같은 default 데이터에서 w1 을 6000스텝 이어 학습 → k=1 단독 / + 보조 (장면 1000~1019) 로 효과만 빨리 본다.
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
D=data/world_default_t
log "[w1b] default wfeat 다시 계산 기다림"
until [ "$(ls $D/episodes/*.wfeat.npy 2>/dev/null | wc -l)" -ge "$(ls $D/episodes/*.npz | wc -l)" ]; do sleep 60; done
sleep 30
INIT=outputs/piper_w1_model/merged
log "[w1b 학습] $INIT 에서 6000스텝 (MemAvailable $(grep MemAvailable /proc/meminfo | awk '{print int($2/1e6)}')GB)"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --log-every 500 --abs-pos --objfeat --feat wfeat --renorm-action --vis-cache cache/vis --frame-stride 6 --sufstate --fp32-master --vlm-lora 16 --vlm-lora-lr 2e-4 --proj-lr 3e-4 --new-lr 3e-4"
$PY train_lora.py --policy $INIT --data $D $TR --batch-size 8 --grad-accum 2 --steps 6000 --lr 3e-5 --eval-every 3000 --save-every 3000 \
    --out outputs/piper_w1b_model 2>&1 | grep --line-buffered -E '에피소드 학습|물체 위치 입력|상태 입력층|새 층|"step": [0-9]*000, "loss"|val_loss|모델 저장|Traceback|Error|Killed'
C=outputs/piper_w1b_model/merged
[ -f $C/config.json ] || { log "학습 실패 — 멈춤"; exit 1; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
FIX="--approach-assist --descend-assist --unjam-assist --yaw-assist --putdown-assist --fresh-held --putdown-near --wrong-guard --mem-refresh --putdown-far --stuck-grasp 40 --yaw-none-reset 10 --mem-tol 0.012 --open-empty --stack-guard --stack-clear --guard-r 0.07 --deliver"
log "[w1b] k=1 단독 ‖ k=1 + 보조"
( $PY piper_sim/world_eval.py --policy $C --seeds 1000-1019 --k 1 --nested $EVQ --out outputs/world_w1b/k1n.json 2>&1 | grep -E "SUMMARY|Traceback|Error" ) &
sleep 60
( $PY piper_sim/world_eval.py --policy $C --seeds 1000-1019 --k 1 --nested $EVQ $FIX --out outputs/world_w1ba/k1n.json 2>&1 | grep -E "SUMMARY|Traceback|Error" ) &
wait
log "끝"
