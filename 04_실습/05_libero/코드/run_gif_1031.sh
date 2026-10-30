#!/usr/bin/env bash
# 10/31 일지용 4K GIF: w10m + 다듬기 + stk 쌓기 성공 장면 (많음 1011 → 안 되면 많음 1017 → 기본 1015). piper-diag2 가 끝난 뒤,
#   자물쇠 + GPU 3개 아래 + 메모리 확인. 끄기는 systemctl --user stop piper-gif1031.
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
RAS="grasp,place,fmask,settle,wxy,gplan,objc,pfine,stk"
CW=outputs/piper_w10m_model/merged
LOCK=outputs/piper/gpu_launch.lock
gpu_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
gpu_n(){ nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -c python; }
mem_gb(){ awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
while systemctl --user is-active --quiet piper-diag2; do sleep 60; done
for cs in many:1011 many:1017 default:1015; do
  c=${cs%%:*}; s=${cs##*:}
  until mkdir $LOCK 2>/dev/null; do sleep 15; done
  while [ "$(gpu_free)" -lt 3400 ] || [ "$(gpu_n)" -ge 3 ] || [ "$(mem_gb)" -lt 6 ]; do sleep 30; done
  log "띄움 4K $c $s"
  PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_vla4k.py --policy $CW --seed $s --k 1 --nested --name world_w10mras_${c}_s$s \
      --label "w10m + 다듬기" $EVQ --refine $RAS > outputs/piper/gif1031_${c}_$s.log 2>&1 &
  P=$!
  sleep 120; rmdir $LOCK
  wait $P
  tail -1 outputs/piper/gif1031_${c}_$s.log
  grep -q "^성공" outputs/piper/gif1031_${c}_$s.log && { log "끝 (성공 장면 $c $s)"; exit 0; }
done
log "끝 (세 장면 모두 실패)"
