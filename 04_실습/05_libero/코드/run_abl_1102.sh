#!/usr/bin/env bash
# 11/2: 물체 많음에서 새 다듬기 RC 가 예전(RA) 과 같음 (29 vs 30/50, 짝 7:6) — 기본·흔든은 +10·+11. 가린 찾기 무시(mocc)가 물체 많은 장면에서
#   손해인지 보려고 w10m + RB (= RC − mocc) 많음 50장면 (outputs/world_w10mrbM_many). w11m 학습이 끝난 뒤 ④ 일꾼들과 같은 자물쇠로 끼어 돈다.
#   끄기는 systemctl --user stop piper-abl1102.
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
LOCK=outputs/piper/gpu_launch.lock
gpu_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
gpu_n(){ nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -c python; }
mem_av(){ awk '/MemAvailable/{print int($2/1e6)}' /proc/meminfo; }
held=0
trap '[ $held -eq 1 ] && rmdir $LOCK 2>/dev/null; exit 0' EXIT TERM INT
until [ -f outputs/piper_w11m_model/merged/worldfeat_v3 ] && grep -q "④" outputs/piper/world11mc.log 2>/dev/null; do sleep 60; done
sleep 600                                            # ④ 일꾼 셋이 먼저 자리 잡게
until mkdir $LOCK 2>/dev/null; do sleep 15; done; held=1
while [ "$(gpu_free)" -lt 3400 ] || [ "$(gpu_n)" -ge 3 ] || [ "$(mem_av)" -lt 6 ]; do sleep 30; done
echo "=== $(date +%m/%d\ %H:%M) 띄움 w10m + RB 많음"
RB="grasp,place,fmask,settle,wxy,gplan,objc,pfine,stk,q2,gq"
PIPER_WORLD=piper_sim/worlds/many.json .venv/bin/python piper_sim/world_eval.py --policy outputs/piper_w10m_model/merged --seeds 1000-1049 --k 1 --nested \
    --latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph --refine $RB --out outputs/world_w10mrbM_many/k1n.json > outputs/piper/w10m_rb_many.log 2>&1 &
P=$!
sleep 120; rmdir $LOCK; held=0
wait $P
echo "=== $(date +%m/%d\ %H:%M) 끝 $(grep SUMMARY outputs/piper/w10m_rb_many.log)"
