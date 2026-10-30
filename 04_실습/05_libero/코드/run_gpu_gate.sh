#!/usr/bin/env bash
# GPU 문지기 (10/30 12:04): 시범 수집(world_collect, CPU 지만 그림 그리기로 GPU 0.25GB씩)이 도는 동안 GPU 작업을 3개로 묶는다.
#   GPU 작업이 3개 이상이면 GPU 자물쇠를 대신 쥐어 새로 못 띄우게, 2개 이하로 줄면 놓고 2분 반 기다린다 (일꾼이 하나 띄우게).
#   수집이 끝나면 자물쇠를 놓고 끝 (그 뒤로는 일꾼의 확인대로 4개까지). 끄기는 systemctl --user stop piper-gpugate.
cd "$(dirname "$0")"
LOCK=outputs/piper/gpu_launch.lock
gpu_n(){ nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .; }
coll(){ ps -eo args | grep -v grep | grep -c "piper_sim/world_collect.py"; }
held=0
trap '[ $held -eq 1 ] && rmdir $LOCK 2>/dev/null' EXIT
while [ "$(coll)" -gt 0 ]; do
  n=$(gpu_n)
  if [ "$n" -ge 3 ]; then
    if [ $held -eq 0 ] && mkdir $LOCK 2>/dev/null; then held=1; echo "=== $(date +%H:%M:%S) 쥠 (GPU 작업 $n)"; fi
  elif [ $held -eq 1 ]; then
    rmdir $LOCK; held=0; echo "=== $(date +%H:%M:%S) 놓음 (GPU 작업 $n)"; sleep 150
  fi
  sleep 2
done
echo "=== $(date +%H:%M:%S) 수집 끝 — 그만"
