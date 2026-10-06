#!/usr/bin/env bash
# When queue v4 reaches DART training, switch to v5 (user order: 100 -> 200 -> 500 -> 1000, then DART).
cd "$(dirname "$0")"
until grep -q "train dart200" logs/queue.log; do sleep 30; done
systemctl --user stop act-queue
sleep 5
rm -rf runs/s1_dart200 runs/s2_dart200 runs/s3_dart200
echo "[$(date '+%m-%d %H:%M:%S')] switched to queue v5 (n500, n1000 before DART)" >> logs/queue.log
systemctl --user reset-failed act-queue 2>/dev/null
systemd-run --user --unit=act-queue -p MemoryMax=9G -p Nice=10 -p IOWeight=50 -p WorkingDirectory=$PWD \
  --setenv=MUJOCO_GL=egl --setenv=MALLOC_MMAP_THRESHOLD_=1048576 /bin/bash $PWD/run_queue5.sh
