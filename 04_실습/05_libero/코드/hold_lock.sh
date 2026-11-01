#!/usr/bin/env bash
# 11/2: GPU 자물쇠를 비는 대로 잡아 쥐고 있는다 (w11mb 가 ⓑ 를 마친 뒤 학습을 바로 띄우지 못하게 — 위치 입력을 바꿔 다시 띄우려고).
#   끄면 놓는다: systemctl --user stop piper-holdlock
cd "$(dirname "$0")"
LOCK=outputs/piper/gpu_launch.lock
held=0
trap '[ $held -eq 1 ] && rmdir $LOCK 2>/dev/null; exit 0' EXIT TERM INT
until mkdir $LOCK 2>/dev/null; do sleep 1; done; held=1
echo "=== $(date +%H:%M:%S) 쥠"
while true; do sleep 5; done
