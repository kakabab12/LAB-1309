#!/usr/bin/env bash
# 일지용 4K GIF 하나 (10/30): 파이프라인과 같은 GPU 자물쇠·메모리 확인을 거쳐 띄운다. 인자: <설정> <장면> <이름> <설명> [world_vla4k 인자...]
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
LOCK=outputs/piper/gpu_launch.lock
gpu_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
gpu_n(){ nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .; }
c=$1 s=$2 n=$3 l=$4; shift 4
until mkdir $LOCK 2>/dev/null; do sleep 15; done
while [ "$(gpu_free)" -lt 3200 ] || [ "$(gpu_n)" -ge 4 ]; do sleep 30; done
echo "=== $(date +%m/%d\ %H:%M) 띄움 GIF $n (GPU 남은 $(gpu_free)MiB)"
PIPER_WORLD=piper_sim/worlds/$c.json .venv/bin/python piper_sim/world_vla4k.py --seed $s --name $n --label "$l" "$@" > outputs/piper/gif_$n.log 2>&1 & P=$!
sleep 120; rmdir $LOCK
wait $P; echo "=== $(date +%m/%d\ %H:%M) 끝 GIF $n"; tail -3 outputs/piper/gif_$n.log
