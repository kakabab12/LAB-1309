#!/usr/bin/env bash
# w2 평가 (10/26 08:50): run_world2_train.sh 의 평가 부분 — piper-world2p 단위가 MemoryMax 10G 라 평가 4줄에서 08:17 OOM 으로 꺼짐.
#   세 설정 × k=1~4 겹침 × (단독, + 보조), 장면 1000~1019. 이미 한 장면은 world_eval 이 건너뜀.
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
C=outputs/piper_w2_model/merged
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
FIX="--approach-assist --descend-assist --unjam-assist --yaw-assist --putdown-assist --fresh-held --putdown-near --wrong-guard --mem-refresh --putdown-far --stuck-grasp 40 --yaw-none-reset 10 --mem-tol 0.012 --open-empty --stack-guard --stack-clear --guard-r 0.07 --deliver"
for mode in alone assist; do
  for cfg in default random many; do
    O=outputs/world_w2_${cfg}; X=""; [ $mode = assist ] && { O=outputs/world_w2a_${cfg}; X="$FIX"; }
    log "[w2 $mode $cfg] k=1~4 겹침 (4줄)"
    for k in 1 2 3 4; do
      ( PIPER_WORLD=piper_sim/worlds/$cfg.json $PY piper_sim/world_eval.py --policy $C --seeds 1000-1019 --k $k --nested $EVQ $X --out $O/k${k}n.json 2>&1 | grep -E "SUMMARY|Traceback|Error" ) &
      sleep 90
    done
    wait
    $PY piper_sim/world_board.py $O --name "w2 $mode $cfg" --md outputs/piper/world_board_w2_${mode}_${cfg}.md --png outputs/media/world_board_w2_${mode}_${cfg}.png | head -12
    $PY piper_sim/world_fail.py $O
  done
done
log "끝"
