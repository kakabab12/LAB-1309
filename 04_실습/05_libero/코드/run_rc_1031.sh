#!/usr/bin/env bash
# 10/31: w10m + 새 다듬기 RC (RA + stk,q2,gq,mocc) 새 기준 일 1개 세 설정 50장면 → outputs/world_w10mrcM_<설정> (w10mraM 과 같은 장면 짝).
#   GPU 는 한 번에 하나만 (w11m DAgger 자리를 남김): 띄운 평가가 끝난 뒤 다음 설정의 자물쇠를 잡는다. 끄기는 systemctl --user stop piper-rc1031.
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
RB="grasp,place,fmask,settle,wxy,gplan,objc,pfine,stk,q2,gq,mocc"
CW=outputs/piper_w10m_model/merged
LOCK=outputs/piper/gpu_launch.lock
gpu_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
gpu_n(){ nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -c python; }
mem_gb(){ awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
held=0
trap '[ $held -eq 1 ] && rmdir $LOCK 2>/dev/null; exit 0' EXIT TERM INT    # 자물쇠를 쥔 채 꺼져도 놓는다
for c in default random many; do
  until mkdir $LOCK 2>/dev/null; do sleep 15; done
  held=1
  while [ "$(gpu_free)" -lt 3400 ] || [ "$(gpu_n)" -ge 3 ] || [ "$(mem_gb)" -lt 6 ]; do sleep 30; done
  log "띄움 w10mrcM $c (GPU 남은 $(gpu_free)MiB, 메모리 $(mem_gb)GB)"
  PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_eval.py --policy $CW --seeds 1000-1049 --k 1 --nested $EVQ --refine $RB \
      --out outputs/world_w10mrcM_$c/k1n.json > outputs/piper/w10m_rc_$c.log 2>&1 &
  P=$!
  sleep 120; rmdir $LOCK; held=0
  wait $P
  log "끝 $c: $(grep SUMMARY outputs/piper/w10m_rc_$c.log)"
done
log "끝"
