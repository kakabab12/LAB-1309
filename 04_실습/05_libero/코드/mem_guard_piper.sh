#!/usr/bin/env bash
# 메모리 감시 (PiPER 수집기용, 10/4) — 남은 메모리가 LIMIT_GB 아래면 가장 최근 수집기(collect_piper.py)를 멈춘다. 수집은 이어서 할 수 있다.
LIMIT_GB=${LIMIT_GB:-4}
while true; do
  avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  if [ "$avail_kb" -lt $((LIMIT_GB * 1024 * 1024)) ]; then
    pid=$(ps -eo pid,etimes,comm,args --sort=etimes | awk '$3=="python" && /collect_piper\.py/ {print $1; exit}')
    [ -n "$pid" ] && { echo "$(date +%H:%M:%S) 남은 메모리 $((avail_kb / 1024))MB — 수집기 $pid 멈춤"; kill "$pid"; }
  fi
  sleep 10
done
