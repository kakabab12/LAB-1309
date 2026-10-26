#!/usr/bin/env bash
# w2 평가 2부 (10/26 13시, 논문용): 새 다듬기(world_refine, 지시문 안 읽음) vs 예전 보조(world_assist, 지시문 읽음), 일 1개.
#   예전 보조 일 2~4개 (default) 는 24장면 중 1개 성공이라 멈춤 — k=1 만 세 설정.
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
C=outputs/piper_w2_model/merged
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
FIX="--approach-assist --descend-assist --unjam-assist --yaw-assist --putdown-assist --fresh-held --putdown-near --wrong-guard --mem-refresh --putdown-far --stuck-grasp 40 --yaw-none-reset 10 --mem-tol 0.012 --open-empty --stack-guard --stack-clear --guard-r 0.07 --deliver"
ev(){ ( PIPER_WORLD=piper_sim/worlds/$1.json $PY piper_sim/world_eval.py --policy $C --seeds 1000-1019 --k 1 --nested $EVQ $3 --out $2/k1n.json 2>&1 | grep -E "SUMMARY|Traceback|Error" | sed "s|^|$2 |" ) & sleep 60; }
log "[1차] 다듬기 default·random·many + 예전 보조 random"
ev default outputs/world_w2r_default "--refine grasp,place"
ev random outputs/world_w2r_random "--refine grasp,place"
ev many outputs/world_w2r_many "--refine grasp,place"
ev random outputs/world_w2a_random "$FIX"
wait
log "[2차] 예전 보조 many + 빼보기 (쥐기만·놓기만, default)"
ev many outputs/world_w2a_many "$FIX"
ev default outputs/world_w2rg_default "--refine grasp"
ev default outputs/world_w2rp_default "--refine place"
wait
$PY piper_sim/world_fail.py outputs/world_w2_default outputs/world_w2a_default outputs/world_w2r_default outputs/world_w2rg_default outputs/world_w2rp_default outputs/world_w2r_random outputs/world_w2a_random outputs/world_w2r_many outputs/world_w2a_many
log "끝"
