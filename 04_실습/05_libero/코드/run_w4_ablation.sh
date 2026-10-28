#!/usr/bin/env bash
# 논문용 빼보기 (10/27 13:55): w4, 기본 설정 일 1개 장면 1000~1049 — 다듬기 쥐기만 / 놓기만 / 예전 보조 (지시문을 읽음).
#   w5 학습과 GPU 를 나눠 쓰므로 한 줄씩 차례로 (GPU 메모리: 학습 7.4GB + 평가 2.2GB)
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
C=outputs/piper_w4_model/merged
FIX="--approach-assist --descend-assist --unjam-assist --yaw-assist --putdown-assist --fresh-held --putdown-near --wrong-guard --mem-refresh --putdown-far --stuck-grasp 40 --yaw-none-reset 10 --mem-tol 0.012 --open-empty --stack-guard --stack-clear --guard-r 0.07 --deliver"
ev(){ PIPER_WORLD=piper_sim/worlds/default.json $PY piper_sim/world_eval.py --policy $C --seeds 1000-1049 --k 1 --nested $EVQ $2 --out $1/k1n.json 2>&1 | grep -E "SUMMARY|Traceback|Error" | sed "s|^|$1 |"; }
log "[빼보기] 쥐기만"; ev outputs/world_w4rg_default "--refine grasp"
log "[빼보기] 놓기만"; ev outputs/world_w4rp_default "--refine place"
log "[예전 보조]"; ev outputs/world_w4a_default "$FIX"
PIPER_WORLD=piper_sim/worlds/default.json $PY piper_sim/world_table.py outputs/world_w4_default outputs/world_w4rg_default outputs/world_w4rp_default outputs/world_w4r_default outputs/world_w4a_default
log "끝"
