#!/usr/bin/env bash
# 10/31 새벽: 지시문 표준 꼴 (--canon, VLA 에 들어가는 지시문만 해석기로 첫 틀로) 를 w10m + 다듬기 새 기준 일 1개 세 설정 50장면에서 잰다
#   (outputs/world_w10mracM_<설정>, w10mraM 과 같은 장면 짝 비교). GPU 는 같은 자물쇠 + 메모리 확인. 끄기는 systemctl --user stop piper-canon1031.
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
RA="grasp,place,fmask,settle,wxy,gplan,objc,pfine"
CW=outputs/piper_w10m_model/merged
LOCK=outputs/piper/gpu_launch.lock
gpu_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
gpu_n(){ nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -c python; }
mem_gb(){ awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
P=()
for c in default random many; do
  until mkdir $LOCK 2>/dev/null; do sleep 15; done
  while [ "$(gpu_free)" -lt 3400 ] || [ "$(gpu_n)" -ge 4 ] || [ "$(mem_gb)" -lt 6 ]; do sleep 30; done
  log "띄움 w10mracM $c (GPU 남은 $(gpu_free)MiB, 메모리 $(mem_gb)GB)"
  ( PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_eval.py --policy $CW --seeds 1000-1049 --k 1 --nested $EVQ --refine $RA --canon \
      --out outputs/world_w10mracM_$c/k1n.json > outputs/piper/w10m_rac_$c.log 2>&1 ) & P+=($!)
  sleep 150; rmdir $LOCK
done
wait "${P[@]}"
grep -h SUMMARY outputs/piper/w10m_rac_*.log
log "끝"
