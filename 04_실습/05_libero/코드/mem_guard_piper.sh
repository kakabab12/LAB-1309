#!/usr/bin/env bash
# 메모리 감시 (10/6 갱신) — 남은 메모리가 LIMIT_GB 아래면 내 작업 중 다시 하면 되는 것부터 멈춘다:
#   검출 미리 계산(objfeat_cache) → 수집기(collect_piper·collect_recovery). 학습·평가는 건드리지 않는다.
#   10/6 14시: 학습(14GB) + 다른 세션 ACT 학습 3개(10GB) + 검출 6개가 겹쳐 0.6GB 까지 떨어진 뒤 프로세스가 모두 죽었다.
LIMIT_GB=${LIMIT_GB:-4}
while true; do
  avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  if [ "$avail_kb" -lt $((LIMIT_GB * 1024 * 1024)) ]; then
    pid=$(ps -eo pid,etimes,args --sort=etimes | awk '/^ *[0-9]+ +[0-9]+ \.venv\/bin\/python (objfeat_cache\.py|piper_sim\/collect_piper\.py|piper_sim\/collect_recovery\.py)/ {print $1; exit}')
    [ -n "$pid" ] && { echo "$(date +%H:%M:%S) 남은 메모리 $((avail_kb / 1024))MB — $pid 멈춤"; kill "$pid"; }
  fi
  sleep 10
done
