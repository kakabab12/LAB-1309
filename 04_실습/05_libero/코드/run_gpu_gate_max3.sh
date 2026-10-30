#!/usr/bin/env bash
# GPU 문지기 (10/31 03:00): 평가 하나가 2.5~2.9GB 로 커져 4개면 11GB 를 넘는다 (10/31 02:26 w10m 혼자 기본이 4번째로 떠서 CUDA OOM).
#   수집과 상관없이 GPU python 작업이 3개 이상이면 GPU 자물쇠를 대신 쥐어 새로 못 띄우게, 2개 이하로 줄면 놓는다.
#   10/31 06:48: 놓은 뒤 2분 반 쉬는 사이 w11m 일꾼 둘이 잇달아 띄워 4개가 됨 (하나를 PID 로 끔) → 쉬지 않고 2초마다 본다
#   (일꾼은 15초마다, 띄운 일꾼은 2분 쥐고 있다 놓으므로 그 뒤에는 문지기가 먼저 잡는다).
#   학습(③) 전에는 다른 GPU 작업이 없어 자물쇠를 쥐지 않는다. 끄기는 systemctl --user stop piper-gpugate3b (쥐고 있으면 놓고 끝).
cd "$(dirname "$0")"
LOCK=outputs/piper/gpu_launch.lock
gpu_n(){ nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -c python; }
held=0
trap '[ $held -eq 1 ] && rmdir $LOCK 2>/dev/null; exit 0' EXIT TERM INT
while true; do
  n=$(gpu_n)
  if [ "$n" -ge 3 ]; then
    if [ $held -eq 0 ] && mkdir $LOCK 2>/dev/null; then held=1; echo "=== $(date +%H:%M:%S) 쥠 (GPU 작업 $n)"; fi
  elif [ $held -eq 1 ]; then
    rmdir $LOCK; held=0; echo "=== $(date +%H:%M:%S) 놓음 (GPU 작업 $n)"
  fi
  sleep 2
done
