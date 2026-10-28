#!/usr/bin/env bash
# w6 (10/27 23시): 팔이 옮길 물체를 가리면 world_detect fallback 이 같은 색 다른 모양을 골라 VLA 위치 입력 기억을 덮었다
#   (w5r many 못 쥔 17장면 중 11). SURE (fallback 찾기는 기억을 안 덮음) 시험 10장면: 물체 기억 오차 16~72cm → 9/10 이 0.1~2.5cm.
#   학습 데이터도 같은 길로 다시 계산 (*.wfeat2s → *.wfeat3s; many 시범 49개 표본에서 쥐기 전 프레임 13% 가 3cm 넘게 달라짐).
#   ① w5 + SURE 평가 (혼자 → 다듬기, 50장면) ‖ 학습 데이터 wfeat2s (CPU 3줄)  ② d4 끝나면 d4 wfeat2s  ③ wfeat3s
#   ④ w6 학습 (w5 에서 15000, + DAgger 4차) ⑤ w6 혼자·다듬기 50장면 ⑥ 표
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
C5=outputs/piper_w5_model/merged
ev(){ ( PIPER_WORLD=piper_sim/worlds/$1.json $PY piper_sim/world_eval.py --policy $2 --seeds $3 --k 1 --nested $EVQ $5 --out $4/k1n.json 2>&1 | grep -E "SUMMARY|Traceback|Error" | sed "s|^|$4 |" ) & sleep 60; }
wf2s(){ for s in 0 1 2; do ( PIPER_WORLD=piper_sim/worlds/$2.json nice -n 10 $PY piper_sim/world_feat_cache.py --data $1 --shard $s --nshard 3 --stride 6 --version 2 --sure 2>&1 | grep -E "^끝|Traceback|Error" ) & done; wait; }
log "[① w5 + SURE 평가 ‖ 학습 데이터 wfeat2s]"
( wf2s data/world_many many; wf2s data/world_random random; wf2s data/world_default_t default
  wf2s data/world_dagger_d1 default; wf2s data/world_dagger_r1 random
  for c in default random many; do wf2s data/world_dagger2_$c $c; wf2s data/world_dagger3_$c $c; done
  log "학습 데이터 wfeat2s 끝" ) &
CACHE=$!
( export PIPER_WF_SURE=1
  for c in default random many; do ev $c $C5 1000-1049 outputs/world_w5s_$c ""; done; wait
  while systemctl --user is-active --quiet piper-d4; do sleep 60; done      # d4 사진 특징 (GPU 2줄) 과 겹치지 않게
  for c in default random many; do ev $c $C5 1000-1049 outputs/world_w5sr_$c "--refine grasp,place"; done; wait
  for c in default random many; do PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_table.py outputs/world_w5_$c outputs/world_w5s_$c outputs/world_w5r_$c outputs/world_w5sr_$c; done
  log "w5 + SURE 평가 끝" ) &
EVALS=$!
log "[②] d4 끝나기 기다림"
while systemctl --user is-active --quiet piper-d4; do sleep 60; done
grep -h STATS outputs/piper/world_dagger4_*.log
wait $CACHE
for c in default random many; do wf2s data/world_dagger4_$c $c; done
D4="data/world_dagger4_default data/world_dagger4_random data/world_dagger4_many"
DATA="data/world_default_t data/world_random data/world_many data/world_dagger_d1 data/world_dagger_r1 data/world_dagger2_default data/world_dagger2_random data/world_dagger2_many data/world_dagger3_default data/world_dagger3_random data/world_dagger3_many $D4"
log "[③ wfeat3s]"
PIPER_WORLD=piper_sim/worlds/default.json $PY piper_sim/world_feat_cache.py --data $DATA --version 3 --from-v2 --sure | tail -1
for d in $DATA; do
  a=$(ls $d/episodes/*.wfeat3s.npy | wc -l); b=$(ls $d/episodes/*.npz | wc -l)
  [ "$a" -eq "$b" ] || { log "$d wfeat3s $a / $b 모자람 — 멈춤"; exit 1; }
done
wait $EVALS
log "[④ w6 학습] $C5 에서 15000스텝, wfeat3s + d4 (MemAvailable $(grep MemAvailable /proc/meminfo | awk '{print int($2/1e6)}')GB)"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --log-every 500 --abs-pos --objfeat --feat wfeat3s --renorm-new-feat --renorm-action --vis-cache cache/vis --frame-stride 6 --sufstate --fp32-master --vlm-lora 16 --vlm-lora-lr 2e-4 --proj-lr 3e-4 --new-lr 3e-4"
$PY train_lora.py --policy $C5 --data $DATA $TR --batch-size 8 --grad-accum 2 --steps 15000 --lr 3e-5 --eval-every 5000 --save-every 5000 \
    --out outputs/piper_w6_model 2>&1 | grep --line-buffered -E '에피소드 학습|새로 쓰는|"step": [0-9]*000, "loss"|val_loss|모델 저장|Traceback|Error|Killed'
C=outputs/piper_w6_model/merged
[ -f $C/worldfeat_sure ] && [ -f $C/worldfeat_v3 ] || { log "w6 모델 표시 없음 — 멈춤"; exit 1; }
log "[⑤ w6] 일 1개 단독 (50장면, 세 설정)"
for c in default random many; do ev $c $C 1000-1049 outputs/world_w6_$c ""; done
wait
log "[⑤ w6] 일 1개 다듬기 (50장면, 세 설정)"
for c in default random many; do ev $c $C 1000-1049 outputs/world_w6r_$c "--refine grasp,place"; done
wait
for c in default random many; do PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_table.py outputs/world_w5s_$c outputs/world_w5sr_$c outputs/world_w6_$c outputs/world_w6r_$c; done
log "끝"
