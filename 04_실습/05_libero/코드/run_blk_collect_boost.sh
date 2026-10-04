#!/usr/bin/env bash
# 정상 수집이 끝나면 전환·재개 수집기를 2개 더 띄워 뒤쪽 장면(2025~2049)부터 모은다 (10/4). 이미 있는 장면은 서로 건너뛴다.
cd "$(dirname "$0")"
export PIPER_BLOCKS=1 MUJOCO_GL=egl
PY=.venv/bin/python; C="piper_sim/collect_piper.py"
while pgrep -f "collect_piper[.]py normal" >/dev/null; do sleep 30; done
echo "=== $(date +%m/%d\ %H:%M) 전환 수집기 2개 추가"
( $PY $C switch --pairs 8:0,8:3,8:5,8:7,8:9,4:5 --episodes 2025-2049 --dart 0.1 --seed 15 --out data/piper_blk_switch > outputs/piper/blk_collect_s3.log 2>&1 ) &
( $PY $C switch --pairs 4:9,1:7,8:1,8:4,1:8,4:1 --episodes 2025-2049 --dart 0.1 --seed 16 --out data/piper_blk_switch > outputs/piper/blk_collect_s4.log 2>&1 ) &
wait
