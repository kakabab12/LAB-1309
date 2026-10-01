#!/usr/bin/env bash
# 메모리 감시 — 남은 메모리가 LIMIT_GB 아래로 내려가면 가장 최근에 시작한 수집기(python collect_v6.py)를 멈춘다.
# 이 PC 는 스왑이 없어 메모리가 다 차면 통째로 멈춘다 (2026-10-01 재부팅). 수집은 이어서 할 수 있으므로 멈춰도 안전하다.
LIMIT_GB=${LIMIT_GB:-5}
while true; do
  avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  if [ "$avail_kb" -lt $((LIMIT_GB * 1024 * 1024)) ]; then
    pid=$(ps -eo pid,etimes,comm,args --sort=etimes | awk '$3=="python" && /collect_v6\.py/ {print $1; exit}')
    if [ -n "$pid" ]; then
      echo "$(date +%H:%M:%S) 남은 메모리 $((avail_kb / 1024))MB — 수집기 $pid 멈춤"
      kill "$pid"
    fi
  fi
  sleep 10
done
