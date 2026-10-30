#!/usr/bin/env bash
# 10/30 새벽: w8 + 다듬기 + 새 부품 (RF, 기본 다 켬), 일 1개 1000~1049 → w8rm 과 같은 장면 비교 (outputs/world_w8${TAG}_설정).
#   GPU 4번째 자리 (남은 메모리 3.5GB 넘을 때만). 위치 입력 계산이 끝나면 ('[②] 끝', w9 학습 전) PID 로 끈다.
#   world_eval 은 끝난 장면을 건너뛰므로 끊겨도 그때까지 잰 장면은 남고, 다시 띄우면 이어서 한다.
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
RF=${RF:-grasp,place,fmask,settle,wxy,gplan,objc,pfine}; TAG=${TAG:-ra}
L=outputs/piper/idle_$TAG.log
log(){ echo "=== $(date +%m/%d\ %H:%M) $*" >> $L; }
stop_now(){ grep -q '\[②\] 끝' outputs/piper/world9_feat.log 2>/dev/null || grep -q '\[④' outputs/piper/world9.log; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
for c in ${CFGS:-default random many}; do
  stop_now && { log "학습 전 — 그만"; break; }
  f=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F, '{print $1-$2}')
  [ "$f" -gt 3000 ] || { log "$c: GPU 남은 ${f}MiB — 안 띄움"; break; }
  PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_eval.py --policy outputs/piper_w8_model/merged --seeds 1000-1049 --k 1 --nested $EVQ \
      --refine $RF --out outputs/world_w8${TAG}_$c/k1n.json > outputs/piper/idle_${TAG}_$c.log 2>&1 &
  P=$!; log "$c 띄움 (PID $P, GPU 남은 ${f}MiB)"
  while kill -0 $P 2>/dev/null; do
    stop_now && { kill -TERM $P; sleep 3; kill -KILL $P 2>/dev/null; log "$c: 학습 전 — 끔"; break 2; }
    sleep 10
  done
  grep SUMMARY outputs/piper/idle_${TAG}_$c.log | tail -1 | sed "s|^|$c |" >> $L
done
log "끝"
