#!/usr/bin/env bash
# piper-w9 (와 그 옆 평가 piper-w9side) 가 끝나면 run_world10.sh 를 piper-w10 유닛으로 띄운다 (GPU 를 겹쳐 쓰지 않게).
cd /home/user/smolVLA
L=outputs/piper/w10_wait.log
echo "=== $(date +%m/%d\ %H:%M) 기다림: piper-w9·piper-w9side 끝" >> $L
while systemctl --user is-active --quiet piper-w9 || systemctl --user is-active --quiet piper-w9side; do sleep 120; done
grep -q '^=== .* 끝$' outputs/piper/world9.log || { echo "=== $(date +%m/%d\ %H:%M) piper-w9 가 끝까지 못 감 — w10 안 띄움" >> $L; exit 0; }
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$n" -eq 0 ] || { echo "=== $(date +%m/%d\ %H:%M) GPU 에 프로세스 $n 개 — 10분 기다림" >> $L; sleep 600; }
systemd-run --user --unit piper-w10 -p MemoryMax=24G -p WorkingDirectory=/home/user/smolVLA bash -c "./run_world10.sh > outputs/piper/world10.log 2>&1"
echo "=== $(date +%m/%d\ %H:%M) piper-w10 띄움" >> $L
