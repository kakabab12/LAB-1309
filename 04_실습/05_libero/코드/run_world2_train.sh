#!/usr/bin/env bash
# w2 (10/25 20시; run_world2_prep.sh 가 사진 특징 뒤에 부름):
#   default·random·many 시범 (새 물체 찾기 wfeat) 로 w1b 에서 이어 30000스텝 (state_proj·state_suffix_proj 학습률 3e-4) → 세 설정 × k=1~4 겹침 × (단독, + 보조) 평가 → 성적표
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
log "[w2] wfeat 다시 계산(piper-wfeat2)·w1b(GPU) 끝나기 기다림"
while systemctl --user is-active --quiet piper-wfeat2 || systemctl --user is-active --quiet piper-w1b || systemctl --user is-active --quiet piper-vis1; do sleep 60; done
# 10/25 17:05 w1b 가 GPU 메모리 부족 (사진 특징 2줄 5GB + 학습 7GB > 11GB) → 사진 특징 shard 1 을 멈췄다. 남은 것 마저 (있는 것은 건너뜀)
B=outputs/piper_np3_model/merged
for s in 0 1; do
  ( $PY vis_cache.py build --policy $B --data data/world_random data/world_many --out cache/vis --batch 8 --stride 6 --shard $s --nshard 2 2>&1 | grep -E "끝|Error|Traceback" ) &
done
wait
# 10/25 21시 DAgger (run_world_dagger.sh, w1b 가 하다 시범 프로그램이 이어받음): 사진 특징이 끝났으니 멈추고 모은 만큼 넣는다
if systemctl --user is-active --quiet piper-dagger; then log "[w2] DAgger 멈춤"; systemctl --user stop piper-dagger; sleep 10; fi
DG=""
for g in data/world_dagger_d1:default data/world_dagger_r1:random; do
  d=${g%%:*}; c=${g##*:}
  [ -d $d/episodes ] || continue
  for f in $(ls -t $d/episodes/*.npz | head -2); do $PY -c "import numpy as np,sys; np.load(sys.argv[1])['action']" $f 2>/dev/null || { log "깨진 파일 지움 $f"; rm -f $f; }; done
  for s in 0 1 2 3; do ( PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_feat_cache.py --data $d --shard $s --nshard 4 --stride 6 | tail -1 ) & done
  wait
  DG="$DG $d"
done
if [ -n "$DG" ]; then
  log "[w2] DAgger 사진 특징 2줄:$DG ($(for d in $DG; do ls $d/episodes/*.npz; done | wc -l) 구간)"
  for s in 0 1; do
    ( $PY vis_cache.py build --policy outputs/piper_np3_model/merged --data $DG --out cache/vis --batch 8 --stride 6 --shard $s --nshard 2 2>&1 | grep -E "끝|Error|Traceback" ) &
  done
  wait
fi
DATA="data/world_default_t data/world_random data/world_many$DG"
for d in $DATA; do
  a=$(ls $d/episodes/*.wfeat.npy | wc -l); b=$(ls $d/episodes/*.npz | wc -l)
  log "$d wfeat $a / $b"
  [ "$a" -eq "$b" ] || { log "wfeat 모자람 — 멈춤"; exit 1; }
done
INIT=outputs/piper_w1b_model/merged      # 10/25 w1b (고친 wfeat + 입력층 학습률): 단독 1/20, + 보조 2/20 — w1 (0/20) 보다 나음
[ -f $INIT/config.json ] || { log "w1b 없음 — 멈춤"; exit 1; }
log "[w2 학습] $INIT 에서 30000스텝 (MemAvailable $(grep MemAvailable /proc/meminfo | awk '{print int($2/1e6)}')GB)"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --log-every 500 --abs-pos --objfeat --feat wfeat --renorm-action --vis-cache cache/vis --frame-stride 6 --sufstate --fp32-master --vlm-lora 16 --vlm-lora-lr 2e-4 --proj-lr 3e-4 --new-lr 3e-4"
$PY train_lora.py --policy $INIT --data $DATA $TR --batch-size 8 --grad-accum 2 --steps 30000 --lr 3e-5 --eval-every 5000 --save-every 5000 \
    --out outputs/piper_w2_model 2>&1 | grep --line-buffered -E '에피소드 학습|VLM LoRA|float32 원본|"step": [0-9]*000, "loss"|val_loss|모델 저장|정책 저장|Traceback|Error|Killed'
C=outputs/piper_w2_model/merged
[ -f $C/config.json ] || { log "학습 실패 — 멈춤"; exit 1; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
FIX="--approach-assist --descend-assist --unjam-assist --yaw-assist --putdown-assist --fresh-held --putdown-near --wrong-guard --mem-refresh --putdown-far --stuck-grasp 40 --yaw-none-reset 10 --mem-tol 0.012 --open-empty --stack-guard --stack-clear --guard-r 0.07 --deliver"
for mode in alone assist; do
  for cfg in default random many; do
    O=outputs/world_w2_${cfg}; X=""; [ $mode = assist ] && { O=outputs/world_w2a_${cfg}; X="$FIX"; }
    log "[w2 $mode $cfg] k=1~4 겹침 (4줄)"
    for k in 1 2 3 4; do
      ( PIPER_WORLD=piper_sim/worlds/$cfg.json $PY piper_sim/world_eval.py --policy $C --seeds 1000-1019 --k $k --nested $EVQ $X --out $O/k${k}n.json 2>&1 | grep -E "SUMMARY|Traceback|Error" ) &
      sleep 90
    done
    wait
    $PY piper_sim/world_board.py $O --name "w2 $mode $cfg" --md outputs/piper/world_board_w2_${mode}_${cfg}.md --png outputs/media/world_board_w2_${mode}_${cfg}.png | head -12
  done
done
log "끝"
