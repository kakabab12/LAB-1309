#!/usr/bin/env bash
# w10m 다듬기 amb 비교 (10/30): run_world10m.sh 가 '끝' 을 찍으면, 같은 장면 일 1개를 다듬기 다 켬 + amb 로 세 설정 잰다
#   (outputs/world_w10mrabM_<설정>). w10mraM 과 짝지어 비교. GPU 는 같은 자물쇠 + 메모리 확인. 끄기는 systemctl --user stop piper-w10mamb.
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
RAB="grasp,place,fmask,settle,wxy,gplan,objc,pfine,amb"
CW=outputs/piper_w10m_model/merged
LOCK=outputs/piper/gpu_launch.lock
gpu_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
gpu_n(){ nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -c python; }
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
until tail -1 outputs/piper/world10m.log | grep -qE "끝$|멈춤"; do sleep 120; done
tail -1 outputs/piper/world10m.log | grep -q "멈춤" && { log "파이프라인이 멈춤 — 그만"; exit 1; }
[ -f $CW/worldfeat_v3 ] || { log "w10m 모델 없음 — 그만"; exit 1; }
P=()
for c in default random many; do
  until mkdir $LOCK 2>/dev/null; do sleep 15; done
  while [ "$(gpu_free)" -lt 3400 ] || [ "$(gpu_n)" -ge 3 ]; do sleep 30; done
  log "띄움 w10mrabM $c (GPU 남은 $(gpu_free)MiB)"
  ( PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_eval.py --policy $CW --seeds 1000-1049 --k 1 --nested $EVQ --refine $RAB \
      --out outputs/world_w10mrabM_$c/k1n.json > outputs/piper/w10m_rab_$c.log 2>&1 ) & P+=($!)
  sleep 150; rmdir $LOCK
done
wait "${P[@]}"
grep -h SUMMARY outputs/piper/w10m_rab_*.log
log "끝"
