#!/usr/bin/env bash
# 11/2: 다듬기 RCX = RC + stk3 (쌓을 물체 반경을 판과 같게 11.4cm) + gq2 (오므리는 중이면 쥐는 자세 계산을 물체 고르는 거리까지)
#   — w11m 으로 세 설정 일 1개 50장면씩 (outputs/world_w11mrxM_*/k1n.json), ④ 의 w11mraM (RC) 와 같은 장면 짝.
#   ④ 시작 15분 뒤부터 ④ 일꾼들과 같은 자물쇠로 한 자리만 차례로. 끄기는 systemctl --user stop piper-rcx1102.
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
RCX="$(cat outputs/piper/ra_choice.txt),stk3,gq2"
for c in default random many; do
  until mkdir $LOCK 2>/dev/null; do sleep 15; done; held=1
  while [ "$(gpu_free)" -lt 3400 ] || [ "$(gpu_n)" -ge 3 ] || [ "$(mem_av)" -lt 6 ]; do sleep 30; done
  echo "=== $(date +%m/%d\ %H:%M) 띄움 w11m + RCX $c"
  PIPER_WORLD=piper_sim/worlds/$c.json .venv/bin/python piper_sim/world_eval.py --policy outputs/piper_w11m_model/merged --seeds 1000-1049 --k 1 --nested \
      --latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph --refine $RCX --out outputs/world_w11mrxM_$c/k1n.json > outputs/piper/w11m_rx_$c.log 2>&1 &
  P=$!
  sleep 120; rmdir $LOCK; held=0
  wait $P
  echo "=== $(date +%m/%d\ %H:%M) 끝 $c $(grep SUMMARY outputs/piper/w11m_rx_$c.log)"
done
