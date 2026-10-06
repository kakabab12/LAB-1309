#!/usr/bin/env bash
cd "$(dirname "$0")"
for s in 1 2 3; do bash gen_stage_v2.sh $s & done
wait
echo "[$(date '+%m-%d %H:%M:%S')] v2 demonstrations complete" >> logs/queue.log
