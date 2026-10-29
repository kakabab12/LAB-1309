#!/usr/bin/env bash
# w9 (10/29 16시): 손목 카메라 몸체가 로봇 가리개 밖으로 나와 '검은 물체'로 잡히던 것 고침
#   (world_detect.RobotMask.extra, install 없이도 손목 카메라 로봇; 백업 outputs/piper/world_detect_before_cammask.py).
#   기본 1000 검은 막대 (w8 + 다듬기): 쥐기 직전 VLA 물체 기억이 10.5cm 튐 → 고친 뒤 손 5cm 안에서 2cm 안, 실패 → 성공 (0.7cm).
#   ① w8 + 고친 가리개 (평가만): 일 1개 다듬기 → 혼자 50장면 (세 설정)
#   ‖ ② 학습 데이터 위치 입력을 고친 가리개로 다시 (*.wfeat2sc → *.wfeat3sc, CPU, nice; 예전 *.wfeat2s 는 그대로)
#   ③ DAgger 6차 (w8 + 고친 가리개, 세 설정 × 200, 12000~12599) → 위치 입력·사진 특징
#   ④ w9 학습 (w8 에서 15000, wfeat3sc, + DAgger 6차) ⑤ w9 일 1개 다듬기·혼자 50장면, 일 2개·일 3개 끼어들기 다듬기 20장면 ⑥ 표
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
C8=outputs/piper_w8_model/merged
EVP=()
ev(){ ( PIPER_WORLD=piper_sim/worlds/$1.json $PY piper_sim/world_eval.py --policy $2 --seeds $3 --k $6 --nested $EVQ $5 --out $4/k${6}n.json 2>&1 | grep -E "SUMMARY|Traceback|Error" | sed "s|^|$4 k$6 |" ) & EVP+=($!); sleep 60; }
evwait(){ wait "${EVP[@]}"; EVP=(); }
wf2c(){ for s in 0 1 2; do ( PIPER_WORLD=piper_sim/worlds/$2.json nice -n 10 $PY piper_sim/world_feat_cache.py --data $1 --shard $s --nshard 3 --stride 6 --version 2 --sure --tag c 2>&1 | grep -E "^끝|Traceback|Error" ) & done; wait; }
D45="data/world_dagger4_default data/world_dagger4_random data/world_dagger4_many data/world_dagger5_default data/world_dagger5_random data/world_dagger5_many"
DN="data/world_daggerN_default data/world_daggerN_random data/world_daggerN_many"
D6="data/world_dagger6_default data/world_dagger6_random data/world_dagger6_many"
DATA="data/world_default_t data/world_random data/world_many data/world_dagger_d1 data/world_dagger_r1 data/world_dagger2_default data/world_dagger2_random data/world_dagger2_many data/world_dagger3_default data/world_dagger3_random data/world_dagger3_many $D45 $DN"

log "[② 학습 데이터 위치 입력 다시 (고친 가리개), CPU 뒤에서]"
( wf2c data/world_many many; wf2c data/world_random random; wf2c data/world_default_t default
  wf2c data/world_dagger_d1 default; wf2c data/world_dagger_r1 random
  for c in default random many; do wf2c data/world_dagger2_$c $c; wf2c data/world_dagger3_$c $c; wf2c data/world_dagger4_$c $c; wf2c data/world_dagger5_$c $c; wf2c data/world_daggerN_$c $c; done
  echo "=== $(date +%m/%d\ %H:%M) [②] 끝" ) > outputs/piper/world9_feat.log 2>&1 &
FP=$!

log "[① w8 + 고친 가리개] 일 1개 다듬기 50장면"
for c in default random many; do ev $c $C8 1000-1049 outputs/world_w8rm_$c "--refine grasp,place" 1; done
evwait
log "[① w8 + 고친 가리개] 일 1개 혼자 50장면"
for c in default random many; do ev $c $C8 1000-1049 outputs/world_w8m_$c "" 1; done
evwait
for c in default random many; do PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_table.py outputs/world_w8_$c outputs/world_w8m_$c outputs/world_w8r_$c outputs/world_w8rm_$c; done

log "[③ DAgger 6차] w8 + 고친 가리개, 세 설정 × 200장면"
DP=()
i=0
for c in default random many; do
  a=$((12000 + i * 200)); b=$((a + 199)); i=$((i + 1))
  ( PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_dagger.py --policy $C8 --seeds $a-$b --t 80 450 --out data/world_dagger6_$c $EVQ > outputs/piper/world_dagger6_$c.log 2>&1 ) &
  DP+=($!); sleep 20
done
wait "${DP[@]}"
grep -h STATS outputs/piper/world_dagger6_*.log
log "[③ DAgger 6차 사진 특징 + 위치 입력]"
VP=()
for s in 0 1; do
  ( $PY vis_cache.py build --policy outputs/piper_np3_model/merged --data $D6 --out cache/vis --batch 8 --stride 6 --shard $s --nshard 2 2>&1 | grep -E "끝|Error|Traceback" ) &
  VP+=($!)
done
( for c in default random many; do wf2c data/world_dagger6_$c $c; done ) > outputs/piper/world9_feat_d6.log 2>&1
wait "${VP[@]}"
log "[②] 끝나기 기다림"
wait $FP
cat outputs/piper/world9_feat.log | grep -c "^끝"
DATA="$DATA $D6"
PIPER_WORLD=piper_sim/worlds/default.json $PY piper_sim/world_feat_cache.py --data $DATA --version 3 --from-v2 --sure --tag c | tail -1
for d in $DATA; do
  a=$(ls $d/episodes/*.wfeat3sc.npy | wc -l); b=$(ls $d/episodes/*.npz | wc -l)
  [ "$a" -eq "$b" ] || { log "$d wfeat3sc $a / $b 모자람 — 멈춤"; exit 1; }
done
$PY - <<'EOF'
import numpy as np, glob
n = d = 0
for f in glob.glob("data/world_*/episodes/*.wfeat2sc.npy"):
    a = np.load(f); b = np.load(f.replace(".wfeat2sc.npy", ".wfeat2s.npy")) if __import__("os").path.exists(f.replace(".wfeat2sc.npy", ".wfeat2s.npy")) else None
    if b is None or a.shape != b.shape: continue
    x = np.abs(a - b).max(axis=1) > 1e-4; n += len(x); d += int(x.sum())
print(f"고친 가리개로 바뀐 위치 입력 프레임 {d} / {n}")
EOF

log "[④ w9 학습] w8 에서 15000스텝, wfeat3sc, + DAgger 6차 (MemAvailable $(grep MemAvailable /proc/meminfo | awk '{print int($2/1e6)}')GB)"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --log-every 500 --abs-pos --objfeat --feat wfeat3sc --renorm-new-feat --renorm-action --vis-cache cache/vis --frame-stride 6 --sufstate --fp32-master --vlm-lora 16 --vlm-lora-lr 2e-4 --proj-lr 3e-4 --new-lr 3e-4"
$PY train_lora.py --policy $C8 --data $DATA $TR --batch-size 8 --grad-accum 2 --steps 15000 --lr 3e-5 --eval-every 5000 --save-every 5000 \
    --out outputs/piper_w9_model 2>&1 | grep --line-buffered -E '에피소드 학습|새로 쓰는|"step": [0-9]*000, "loss"|val_loss|모델 저장|Traceback|Error|Killed'
C=outputs/piper_w9_model/merged
[ -f $C/worldfeat_sure ] && [ -f $C/worldfeat_v3 ] && [ -f $C/worldfeat_cammask ] || { log "w9 모델 표시 없음 — 멈춤"; exit 1; }

log "[⑤ w9] 일 1개 다듬기 50장면"
for c in default random many; do ev $c $C 1000-1049 outputs/world_w9r_$c "--refine grasp,place" 1; done
evwait
log "[⑤ w9] 일 1개 혼자 50장면"
for c in default random many; do ev $c $C 1000-1049 outputs/world_w9_$c "" 1; done
evwait
log "[⑤ w9] 일 2개 끼어들기 다듬기 20장면"
for c in default random many; do ev $c $C 1000-1019 outputs/world_w9r_$c "--refine grasp,place" 2; done
evwait
log "[⑤ w9] 일 3개 끼어들기 다듬기 20장면 (물체 N개 목표의 첫 기준)"
for c in default random many; do ev $c $C 1000-1019 outputs/world_w9r_$c "--refine grasp,place" 3; done
evwait
for c in default random many; do PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_table.py outputs/world_w8m_$c outputs/world_w8rm_$c outputs/world_w9_$c outputs/world_w9r_$c; done
log "끝"
