#!/usr/bin/env bash
# 10/31 새벽: (1) 다듬기 stk (쌓을 물체도 판처럼 크기 + 여유 안) 를 w10m 새 기준 일 1개 쌓기 장면 33개에서 잰다 (w10mraM 과 짝 비교)
#   (2) w10m + 다듬기가 지시한 물체 3cm 안까지 가고도 안 쥔 장면·판 옆에서 놓친 장면을 다시 돌려 스텝마다 기록 (world_trace).
#   GPU 는 다른 일꾼과 같은 자물쇠 + 남은 메모리 확인 (수집 중에는 문지기가 3개로 묶음). 끄기는 systemctl --user stop piper-diag1031.
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
take(){   # 자물쇠를 쥐고 GPU·메모리가 빌 때까지 (띄운 뒤 2분 반 뒤에 놓는다)
  until mkdir $LOCK 2>/dev/null; do sleep 15; done
  while [ "$(gpu_free)" -lt 3400 ] || [ "$(gpu_n)" -ge 4 ] || [ "$(mem_gb)" -lt 6 ]; do sleep 30; done
}
declare -A STK=([default]=1006,1010,1013,1015,1021,1025,1028,1032,1035,1042,1043,1047
                [random]=1006,1010,1013,1015,1021,1025,1028,1032,1035,1042,1043,1047
                [many]=1009,1010,1011,1012,1016,1017,1027,1028,1030)
declare -A TR=([default]=1001,1005,1026 [random]=1004,1014,1017,1019 [many]=1001)
P=()
for c in default random many; do
  take
  log "띄움 w10mrasM $c (쌓기 ${STK[$c]}) GPU 남은 $(gpu_free)MiB, 메모리 $(mem_gb)GB"
  ( PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_eval.py --policy $CW --seeds ${STK[$c]} --k 1 --nested $EVQ --refine $RA,stk \
      --out outputs/world_w10mrasM_$c/k1s.json > outputs/piper/w10m_ras_$c.log 2>&1 ) & P+=($!)
  sleep 150; rmdir $LOCK
done
( for c in default random many; do
    take
    log "띄움 따라가기 $c (${TR[$c]}) GPU 남은 $(gpu_free)MiB"
    PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_trace.py --policy $CW --seeds ${TR[$c]} --k 1 --nested $EVQ --refine $RA \
        --name tr_w10mra_$c > outputs/piper/trace_$c.log 2>&1 &
    TP=$!
    sleep 150; rmdir $LOCK
    wait $TP
    log "끝 따라가기 $c: $(grep -c "→" outputs/piper/trace_$c.log)개"
  done ) & P+=($!)
wait "${P[@]}"
grep -h SUMMARY outputs/piper/w10m_ras_*.log
log "끝"
