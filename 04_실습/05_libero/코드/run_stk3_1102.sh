#!/usr/bin/env bash
# 11/2: 다듬기 stk3 (쌓을 물체 반경을 판과 같게 11.4cm) — w10m + RC + stk3 쌓기 33장면 (기본·흔든 12, 많음 9) vs RC 같은 장면.
#   w11m 학습이 끝난 뒤 ④ 일꾼들과 같은 자물쇠로 한 자리만 차례로. 끄기는 systemctl --user stop piper-stk31102.
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
LOCK=outputs/piper/gpu_launch.lock
gpu_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
gpu_n(){ nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -c python; }
mem_av(){ awk '/MemAvailable/{print int($2/1e6)}' /proc/meminfo; }
held=0
trap '[ $held -eq 1 ] && rmdir $LOCK 2>/dev/null; exit 0' EXIT TERM INT
until [ -f outputs/piper_w11m_model/merged/worldfeat_v3 ] && grep -q "④" outputs/piper/world11mc.log 2>/dev/null; do sleep 60; done
sleep 900
RC3="$(cat outputs/piper/ra_choice.txt),stk3"
declare -A STK=([default]=1006,1010,1013,1015,1021,1025,1028,1032,1035,1042,1043,1047
                [random]=1006,1010,1013,1015,1021,1025,1028,1032,1035,1042,1043,1047
                [many]=1009,1010,1011,1012,1016,1017,1027,1028,1030)
for c in default random many; do
  until mkdir $LOCK 2>/dev/null; do sleep 15; done; held=1
  while [ "$(gpu_free)" -lt 3400 ] || [ "$(gpu_n)" -ge 3 ] || [ "$(mem_av)" -lt 6 ]; do sleep 30; done
  echo "=== $(date +%m/%d\ %H:%M) 띄움 w10m + RC + stk3 $c"
  PIPER_WORLD=piper_sim/worlds/$c.json .venv/bin/python piper_sim/world_eval.py --policy outputs/piper_w10m_model/merged --seeds ${STK[$c]} --k 1 --nested \
      --latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph --refine $RC3 --out outputs/world_w10mrc3M_$c/k1s.json > outputs/piper/w10m_rc3_$c.log 2>&1 &
  P=$!
  sleep 120; rmdir $LOCK; held=0
  wait $P
  echo "=== $(date +%m/%d\ %H:%M) 끝 $c $(grep SUMMARY outputs/piper/w10m_rc3_$c.log)"
done
