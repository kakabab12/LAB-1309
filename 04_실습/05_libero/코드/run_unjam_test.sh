#!/usr/bin/env bash
# 막힘 풀기 시험 (10/21 18시): np1 + 보조 '블록 위에서 못 쥠' 장면을 그대로 다시 돌려 고쳐지나 본다 (같은 장면 = 같은 결과가 나오는 것 확인함)
#   v1 = 보조(AS) + 막힘 풀기 / v2 = + 손목 맞추기(6번 관절 한계 봄). GPU 는 학습 옆이라 하나씩 (남은 메모리 3.5GB 넘을 때)
cd "$(dirname "$0")"
export PIPER_NOPED=1 PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl
PY=.venv/bin/python
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
AS="--approach-assist --descend-assist"
O=outputs/piper_unjam_test
for it in 1:7:1001 4:5:1003 8:5:1017 4:9:1007 8:0:1007 8:7:1012 1:7:1012 8:7:1013 4:9:1015; do
  IFS=: read a b ep <<< "$it"
  for v in v1 v2; do
    opt="$AS --unjam-assist"; [ $v = v2 ] && opt="$opt --yaw-assist"
    while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 3500 ]; do sleep 30; done
    $PY piper_sim/eval_piper.py --policy outputs/piper_np1_model/merged --task-a $a --task-b $b --switch-at grasp:3 --strategy keep \
      --episodes 1 --start-episode $ep $EVQ $opt --out $O/${v}_A${a}B${b}_$ep 2>&1 | grep -E "Traceback|Error" | head -3
    echo "$(date +%H:%M) 끝 $v A${a}B${b} $ep"
  done
done
echo 모두끝
