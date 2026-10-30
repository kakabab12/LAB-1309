#!/usr/bin/env bash
# 10/31 04시: 새 다듬기 (RA + stk + q2 + gq) 로 따라가기 8장면을 다시 (같은 장면 → 고침이 붙는지). 설정마다 한 번에 하나,
#   띄운 작업이 끝난 뒤 다음 자물쇠 (다른 일꾼 자리 뺏지 않게). 끄기는 systemctl --user stop piper-diag2.
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
RB="grasp,place,fmask,settle,wxy,gplan,objc,pfine,stk,q2,gq"
CW=outputs/piper_w10m_model/merged
LOCK=outputs/piper/gpu_launch.lock
gpu_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
gpu_n(){ nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -c python; }
mem_gb(){ awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
declare -A TR=([default]=1001,1005,1026 [random]=1004,1014,1017,1019 [many]=1001)
for c in default random many; do
  until mkdir $LOCK 2>/dev/null; do sleep 15; done
  while [ "$(gpu_free)" -lt 3400 ] || [ "$(gpu_n)" -ge 3 ] || [ "$(mem_gb)" -lt 6 ]; do sleep 30; done
  log "띄움 새 다듬기 따라가기 $c (${TR[$c]}) GPU 남은 $(gpu_free)MiB"
  PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_trace.py --policy $CW --seeds ${TR[$c]} --k 1 --nested $EVQ --refine $RB \
      --name tr_w10mrb_$c > outputs/piper/trace2_$c.log 2>&1 &
  TP=$!
  sleep 120; rmdir $LOCK
  wait $TP
  log "끝 $c: $(grep -h "→" outputs/piper/trace2_$c.log | cut -c1-60 | tr '\n' ' ')"
done
log "끝"
