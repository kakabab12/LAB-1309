#!/usr/bin/env bash
# 여러 모양·N개 장면 1라운드 (10/24 사용자 "객체도 여러모양으로 n개씩 … 무제한", "360도", "하드코딩 아닌걸로", "모든 변수 환경 95%")
#   ① 팔 닿는 범위 지도(planner-reach) 기다림 ② 시범 3줄 (장면 3000~4199, 일 1~4개, 70% 끼어들기 겹침, piper_sim/world_collect.py)
#   ③ 다듬기·물체 위치 입력(wfeat)·사진 특징 ④ piper-np5 끝나면 w1 학습 (np5v 또는 np5f 중 단독 혼자 점수가 높은 것에서,
#      --objfeat --feat wfeat --reset-feat --renorm-action --fp32-master --vlm-lora 16, 20000스텝) ⑤ w1 단독 평가 k=1·2·3·4 겹침 (장면 1000~1019)
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
D=data/world_default
log "[world] 팔 닿는 범위 지도 기다림"
while systemctl --user is-active --quiet planner-reach; do sleep 60; done
log "[world 시범 수집] 3줄"
for j in 0 1 2; do
  a=$((3000 + j * 400)); b=$((a + 399))
  ( $PY piper_sim/world_collect.py --seeds $a-$b --k 1 4 --nested-frac 0.7 --dart 0.1 --out $D > outputs/piper/world_collect_$j.log 2>&1 ) &
  sleep 20
done
wait
grep -h STATS outputs/piper/world_collect_*.log
log "수집 끝: 구간 $(ls $D/episodes/*.npz | wc -l)"
rm -rf ${D}_t; $PY piper_sim/trim_stalls.py $D ${D}_t | tail -1
for s in 0 1 2; do ( $PY piper_sim/world_feat_cache.py --data ${D}_t --shard $s --nshard 3 | tail -1 ) & done; wait
log "[w1] piper-np5 끝나기 기다림 (GPU·메모리)"
while systemctl --user is-active --quiet piper-np5; do sleep 120; done
B=outputs/piper_np3_model/merged
( $PY vis_cache.py build --policy $B --data ${D}_t --out cache/vis --batch 8 --stride 3 --shard 0 --nshard 2 2>&1 | grep -E "끝|Error|Traceback" ) &
( $PY vis_cache.py build --policy $B --data ${D}_t --out cache/vis --batch 8 --stride 3 --shard 1 --nshard 2 2>&1 | grep -E "끝|Error|Traceback" ) &
wait
score(){ $PY - "$1" <<'EOF'
import json, glob, sys
fs = glob.glob(f"outputs/{sys.argv[1]}_forget/*.json")
print(sum(e["a_success"] for f in fs for e in json.load(open(f))["episodes"]) if len(fs) == 10 else -1)
EOF
}
SF=$(score piper_np5fh); SV=$(score piper_np5vh)
INIT=outputs/piper_np5f_model/merged; [ "$SV" -gt "$SF" ] && INIT=outputs/piper_np5v_model/merged
log "[w1 학습] 시작 모델 $INIT (단독 혼자 np5f $SF / np5v $SV), 20000스텝 (MemAvailable $(grep MemAvailable /proc/meminfo | awk '{print int($2/1e6)}')GB)"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --log-every 500 --abs-pos --objfeat --feat wfeat --reset-feat --renorm-action --vis-cache cache/vis --frame-stride 3 --sufstate --fp32-master --vlm-lora 16 --vlm-lora-lr 2e-4"
$PY train_lora.py --policy $INIT --data ${D}_t $TR --batch-size 8 --grad-accum 2 --steps 20000 --lr 3e-5 --eval-every 5000 --save-every 5000 \
    --out outputs/piper_w1_model 2>&1 | grep --line-buffered -E '에피소드 학습|VLM LoRA|float32 원본|"step": [0-9]*000, "loss"|val_loss|모델 저장|정책 저장|Traceback|Error|Killed'
C=outputs/piper_w1_model/merged
[ -f $C/config.json ] || { log "학습 실패 — 멈춤"; exit 1; }
log "[w1 단독 평가] k=1·2·3·4 겹침, 장면 1000~1019 (4줄)"
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
for k in 1 2 3 4; do
  ( $PY piper_sim/world_eval.py --policy $C --seeds 1000-1019 --k $k --nested $EVQ --out outputs/world_w1/k${k}n.json 2>&1 | grep -E "SUMMARY|Traceback|Error" ) &
  sleep 90
done
wait
log "끝"
