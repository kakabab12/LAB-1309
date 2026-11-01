#!/usr/bin/env bash
# 11/2 시험: VLA 위치 입력에서 쥔 물체 기억이 손을 따라가게 (PIPER_WF_HELD=1, world_feat 설명) — 다시 학습 없이 w10m + RC 기본 쌓기 12장면.
#   같은 장면 RC 는 5/12 (outputs/world_w10mrcM_default). GPU 자물쇠 + 3개 미만. 끄기는 systemctl --user stop piper-held1102.
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
LOCK=outputs/piper/gpu_launch.lock
gpu_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
gpu_n(){ nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -c python; }
held=0
trap '[ $held -eq 1 ] && rmdir $LOCK 2>/dev/null; exit 0' EXIT TERM INT
until mkdir $LOCK 2>/dev/null; do sleep 15; done; held=1
while [ "$(gpu_free)" -lt 3400 ] || [ "$(gpu_n)" -ge 3 ]; do sleep 30; done
echo "=== $(date +%H:%M) 띄움"
PIPER_WF_HELD=1 PIPER_WORLD=piper_sim/worlds/${CFG:-default}.json .venv/bin/python piper_sim/world_eval.py --policy outputs/piper_w10m_model/merged \
    --seeds ${SEEDS:-1006,1010,1013,1015,1021,1025,1028,1032,1035,1042,1043,1047} --k 1 --nested --latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph \
    --refine $(cat outputs/piper/ra_choice.txt) --out outputs/world_w10mrchM_${CFG:-default}/k1s.json &
P=$!
sleep 120; rmdir $LOCK; held=0
wait $P
echo "=== $(date +%H:%M) 끝"
