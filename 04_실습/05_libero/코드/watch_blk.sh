#!/usr/bin/env bash
# 블록 실험 감시 (10/5): 2분마다 상태를 outputs/piper/STATUS.txt 에 쓰고, 문제가 있으면 outputs/piper/watchdog.log 에 한 줄.
#   멈춤: 라운드 스크립트는 살아 있는데 학습·평가·수집 프로세스가 6분 넘게 0개 / 작업이 있는데 GPU 가 20분 넘게 0%
#   ⚠️ 프로세스는 명령줄 맨 앞(^.venv/bin/python, ^bash ./run_)으로만 센다 — 10/5 새벽 pgrep 이 다른 셸의 스크립트 본문에 걸려 9시간 멈춘 일
cd "$(dirname "$0")"
W=outputs/piper/watchdog.log; S=outputs/piper/STATUS.txt
idle=0; gzero=0; done_said=0
while true; do
  now=$(date +%m/%d\ %H:%M)
  jobs=$(ps -eo args | grep -cE "^\.venv/bin/python (train_lora|piper_sim/|a2c2|vis_cache)")
  rounds=$(ps -eo args | grep -cE "^bash \./run_(blk|piper)")
  gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  mem=$(awk '/MemAvailable/ {printf "%.1f", $2/1048576}' /proc/meminfo)
  {
    echo "$now  작업 $jobs  순서 스크립트 $rounds  GPU ${gpu}%  남은 메모리 ${mem}GB"
    ps -eo etime,args | grep -E "^ *[0-9:-]+ (\.venv/bin/python|bash \./run_)" | sed -E 's/--(data|policy) [^ ]+//' | cut -c1-150
    for m in piper_b1c piper_b2 piper_b3 piper_b4; do
      [ -f outputs/${m}_model/history.json ] && echo "$m 학습: $(python3 -c "import json; h=json.load(open('outputs/${m}_model/history.json')); print(h[-1])")"
    done
  } > $S
  if [ "$jobs" -eq 0 ] && [ "$rounds" -gt 0 ]; then idle=$((idle + 1)); else idle=0; fi
  [ $idle -eq 3 ] && echo "$now STALL 순서 스크립트는 살아 있는데 작업이 6분째 0개" >> $W
  if [ "$jobs" -gt 0 ] && [ "${gpu:-0}" -eq 0 ]; then gzero=$((gzero + 1)); else gzero=0; fi
  [ $gzero -eq 10 ] && echo "$now STALL 작업이 있는데 GPU 20분째 0%" >> $W
  if [ "$jobs" -eq 0 ] && [ "$rounds" -eq 0 ]; then
    [ $done_said -eq 0 ] && echo "$now IDLE 모든 작업·순서가 끝남 — 다음 단계를 걸어야 함" >> $W; done_said=1
  else done_said=0; fi
  python3 -c "import sys; sys.exit(0 if float('$mem') < 3.0 else 1)" && echo "$now MEM 남은 메모리 ${mem}GB" >> $W
  sleep 120
done
