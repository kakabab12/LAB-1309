#!/usr/bin/env bash
# 11/2 0시: w11m 파이프라인 (run_world11m.sh) 이 10/31 10:41 ② 에서 멈춘 것을 이어서.
#   멈춘 까닭: world_feat_cache 세 줄이 같이 시작하며 손목 카메라 로봇 파일(piper_noped_cam_random/robot.xml)을 다시 쓰다 다른 줄이 빈 파일을 읽음
#   → daggerW3_random 사진 위치 입력 15/47 에서 끊기고 3판 만들기가 거기서 멈춤 (world.make_cam_robot 을 바뀐 때만·바꿔 끼우기로 고침).
#   ⓐ (CPU) daggerW3_random 2판 나머지 → 새 데이터 3판 → 개수 확인
#   ⓑ (GPU 둘) 그동안 w10m + RC 다듬기 흔든(1005~ 이어서)·많음 50장면 (outputs/world_w10mrcM_*, 10/31 에 멈춘 것)
#   ⓒ 둘 다 끝나면 ③ w11m 학습 (학습 동안 자물쇠를 쥐어 다른 GPU 작업을 막음) → ④ 평가 12항목 (GPU 일꾼 셋, RC 다듬기)
#   끄기는 systemctl --user stop piper-w11mb 로만 (쥐고 있던 자물쇠는 놓고 끝).
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
RC=$(cat outputs/piper/ra_choice.txt)
CD=outputs/piper_w10m_model/merged
LOCK=outputs/piper/gpu_launch.lock
CL=outputs/piper/claim_w11m
gpu_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
gpu_n(){ nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -c python; }
mem_av(){ awk '/MemAvailable/{print int($2/1e6)}' /proc/meminfo; }
held=0
trap '[ $held -eq 1 ] && rmdir $LOCK 2>/dev/null; exit 0' EXIT TERM INT
take(){ until mkdir $LOCK 2>/dev/null; do sleep 15; done; held=1
  while [ "$(gpu_free)" -lt 3400 ] || [ "$(gpu_n)" -ge 3 ] || [ "$(mem_av)" -lt 6 ]; do sleep 30; done; }
give(){ rmdir $LOCK; held=0; }

NEW="data/world_motion2_default data/world_motion2_random data/world_motion2_many data/world_daggerW1_default data/world_daggerW1_random data/world_daggerW1_many
     data/world_daggerW2_default data/world_daggerW2_random data/world_daggerW2_many data/world_daggerW3_default data/world_daggerW3_random data/world_daggerW3_many"
D45="data/world_dagger4_default data/world_dagger4_random data/world_dagger4_many data/world_dagger5_default data/world_dagger5_random data/world_dagger5_many"
DN="data/world_daggerN_default data/world_daggerN_random data/world_daggerN_many"
D6="data/world_dagger6_default data/world_dagger6_random data/world_dagger6_many"
N10="data/world_motion_default data/world_motion_random data/world_motion_many data/world_daggerM2_default data/world_daggerM2_random data/world_daggerM2_many
     data/world_daggerM3_default data/world_daggerM3_random data/world_daggerM3_many data/world_daggerK3_default"
DATA="data/world_default_t data/world_random data/world_many data/world_dagger_d1 data/world_dagger_r1 data/world_dagger2_default data/world_dagger2_random data/world_dagger2_many data/world_dagger3_default data/world_dagger3_random data/world_dagger3_many $D45 $DN $D6 $N10 $NEW"

( log "ⓐ daggerW3_random 2판 $(ls data/world_daggerW3_random/episodes/*.wfeat2sc.npy | wc -l) / $(ls data/world_daggerW3_random/episodes/*.npz | wc -l)"
  FP=(); for s in 0 1 2; do ( PIPER_WORLD=piper_sim/worlds/random.json nice -n 10 $PY piper_sim/world_feat_cache.py --data data/world_daggerW3_random \
      --shard $s --nshard 3 --stride 6 --version 2 --sure --tag c 2>&1 | grep -E "^끝|Traceback|Error" ) & FP+=($!); sleep 20; done; wait "${FP[@]}"
  PIPER_WORLD=piper_sim/worlds/default.json $PY piper_sim/world_feat_cache.py --data $NEW --version 3 --from-v2 --sure --tag c | tail -1
  log "ⓐ 끝" ) &
PA=$!

PB=()
for c in random many; do
  take
  log "ⓑ 띄움 w10mrcM $c (GPU 남은 $(gpu_free)MiB, 메모리 $(mem_av)GB)"
  PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_eval.py --policy $CD --seeds 1000-1049 --k 1 --nested $EVQ --refine $RC \
      --out outputs/world_w10mrcM_$c/k1n.json >> outputs/piper/w10m_rc_$c.log 2>&1 &
  PB+=($!)
  sleep 120; give
done
wait $PA
for d in $DATA; do
  a=$(ls $d/episodes/*.wfeat3sc.npy | wc -l); b=$(ls $d/episodes/*.npz | wc -l)
  [ "$a" -eq "$b" ] || { log "$d wfeat3sc $a / $b 모자람 — 멈춤 (ⓑ 는 끝까지)"; wait "${PB[@]}"; exit 1; }
done
wait "${PB[@]}"
grep -h SUMMARY outputs/piper/w10m_rc_*.log
log "ⓑ 끝"

take
until [ "$(gpu_n)" -eq 0 ]; do sleep 30; done
log "[③ w11m 학습] $CD 에서 15000스텝, + 새 데이터 (w10m DAgger·시범), 지시문 바꿔 쓰기 50% (MemAvailable $(mem_av)GB) — 자물쇠 쥠"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --log-every 500 --abs-pos --objfeat --feat wfeat3sc --renorm-new-feat --renorm-action --vis-cache cache/vis --frame-stride 6 --sufstate --fp32-master --vlm-lora 16 --vlm-lora-lr 2e-4 --proj-lr 3e-4 --new-lr 3e-4"
PIPER_WORLD=piper_sim/worlds/default.json $PY train_lora.py --policy $CD --data $DATA $TR --instr-aug 0.5 --batch-size 8 --grad-accum 2 --steps 15000 --lr 3e-5 --eval-every 5000 --save-every 5000 \
    --out outputs/piper_w11m_model 2>&1 | grep --line-buffered -E '에피소드 학습|새로 쓰는|"step": [0-9]*000, "loss"|val_loss|모델 저장|Traceback|Error|Killed'
give
CW=outputs/piper_w11m_model/merged
[ -f $CW/worldfeat_sure ] && [ -f $CW/worldfeat_v3 ] && [ -f $CW/worldfeat_cammask ] || { log "w11m 모델 표시 없음 — 멈춤"; exit 1; }

run_item(){ IFS=: read -r kind c a b d e f <<< "$1"
  local opt=""; [ "$d" != "-" ] && opt="--refine $d"
  env PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_eval.py --policy $f --seeds $b --k $a --nested $EVQ $opt \
      --out outputs/world_${e}_$c/k${a}n.json > outputs/piper/w11m_${e}_${c}_k$a.log 2>&1; }
gworker(){ for it in "$@"; do mkdir $CL/${it//[:\/,]/_} 2>/dev/null || continue
    until mkdir $LOCK 2>/dev/null; do sleep 15; done
    while [ "$(gpu_free)" -lt 3400 ] || [ "$(gpu_n)" -ge 3 ] || [ "$(mem_av)" -lt 6 ]; do sleep 30; done
    log "띄움 $it (GPU 남은 $(gpu_free)MiB, 프로세스 $(gpu_n), 메모리 $(mem_av)GB)"
    run_item "$it" & local P=$!
    sleep 120; rmdir $LOCK
    wait $P; log "끝 $it"
  done; }
log "[④ w11m 새 기준 평가] 다듬기 $RC (GPU 일꾼 셋)"
E4="ev:default:1:1000-1049:$RC:w11mraM:$CW ev:random:1:1000-1049:$RC:w11mraM:$CW ev:many:1:1000-1049:$RC:w11mraM:$CW
    ev:default:2:1000-1019:$RC:w11mraM:$CW ev:default:1:1000-1049:-:w11mM:$CW ev:random:1:1000-1049:-:w11mM:$CW ev:many:1:1000-1049:-:w11mM:$CW
    ev:random:2:1000-1019:$RC:w11mraM:$CW ev:many:2:1000-1019:$RC:w11mraM:$CW
    ev:default:3:1000-1019:$RC:w11mraM:$CW ev:random:3:1000-1019:$RC:w11mraM:$CW ev:many:3:1000-1019:$RC:w11mraM:$CW"
W=()
for i in 1 2 3; do gworker $E4 & W+=($!); sleep 5; done
wait "${W[@]}"
grep -h SUMMARY outputs/piper/w11m_w11m*_k*.log
for c in default random many; do PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_table.py outputs/world_w10mM_$c outputs/world_w10mraM_$c outputs/world_w11mM_$c outputs/world_w11mraM_$c; done
log "끝"
