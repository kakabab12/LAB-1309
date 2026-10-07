#!/usr/bin/env bash
# 작은 블록 시험 (10/7): 다시 학습하지 않은 bo1 을 3.5cm 블록 장면에 그대로 넣어 본다 + 4.5cm 에서 실패 분류 기록.
#   bo1 실패의 대부분이 '쥐자마자 미끄러짐' 이고, 손가락이 1.4~1.5cm 어긋나 블록 윗면에 걸린 것이었다 (4장면 진단).
#   그리퍼 최대 7cm → 4.5cm 블록은 한쪽 여유 1.25cm, 3.5cm 면 1.75cm. 이 가설을 학습 없이 먼저 잰다.
set -u
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 MUJOCO_GL=egl
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 11 --ttrtc --n-action-steps 1"
B=outputs/piper_bo1_model/merged
ngpu(){ nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -vc rustdesk; }   # 원격 접속 프로그램은 빼고
mem(){ awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
log "GPU 자리 기다림 (지금 $(ngpu)개)"
until [ "$(ngpu)" -le 2 ] && [ "$(mem)" -ge 8 ]; do sleep 60; done
log "시작 (GPU $(ngpu)개, 남은 메모리 $(mem)GB)"
ev(){ local o=$1; shift; for t in "$@"; do $PY piper_sim/eval_piper.py --policy $B --task-a $t --strategy none --episodes 20 --start-episode 1000 $EVQ --out outputs/${o}_forget 2>&1 | grep -E "$F"; done; }
( PIPER_CUBE=0.035 ev piper_bo1c35 0 8 2 ) & P1=$!
sleep 30
if [ "$(ngpu)" -le 3 ]; then ( ev piper_bo1L 0 8 2 ) & P2=$!; wait $P2; fi
wait $P1
log "끝"
