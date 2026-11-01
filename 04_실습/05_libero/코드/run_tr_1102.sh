#!/usr/bin/env bash
# 11/2: w10m + RC 기본 실패 셋 따라가기 (쌓기에서 목표 10~11cm 옆에 내려놓음 1021·1028, 목표 물체를 먼저 집음 1006) — VLA 위치 입력(wf)까지 기록.
#   GPU 자물쇠를 쥐고 3개 미만일 때 띄우고, 2분 뒤 놓는다. 끄기는 systemctl --user stop piper-tr1102.
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
LOCK=outputs/piper/gpu_launch.lock
gpu_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
gpu_n(){ nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -c python; }
held=0
trap '[ $held -eq 1 ] && rmdir $LOCK 2>/dev/null; exit 0' EXIT TERM INT
until mkdir $LOCK 2>/dev/null; do sleep 15; done; held=1
while [ "$(gpu_free)" -lt 3400 ] || [ "$(gpu_n)" -ge 3 ]; do sleep 30; done
echo "=== $(date +%H:%M) 띄움 (GPU 남은 $(gpu_free)MiB)"
PIPER_WORLD=piper_sim/worlds/default.json .venv/bin/python piper_sim/world_trace.py --policy outputs/piper_w10m_model/merged --seeds ${SEEDS:-1006,1021,1028} \
    --k 1 --nested --name tr_w10mrc_default --refine $(cat outputs/piper/ra_choice.txt) --latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph &
P=$!
sleep 120; rmdir $LOCK; held=0
wait $P
echo "=== $(date +%H:%M) 끝"
