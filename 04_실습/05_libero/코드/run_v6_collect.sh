#!/usr/bin/env bash
# 6차 데이터 수집 — 정상 10개 태스크 + 전환·재개 27쌍. 모델 없이 CPU 만 쓴다. 이미 모은 것은 건너뛴다.
# ⚠️ 수집기 하나가 메모리 2.4GB 를 쓴다. 이 PC 는 31GB, 스왑 0 이라 16개를 한꺼번에 돌리자 멈춰 재부팅됐다
#    (2026-10-01 15:5x). 동시에 최대 MAXP 개만 돌린다.
cd "$(dirname "$0")"
MAXP=${MAXP:-7}
mkdir -p outputs/v6
{
  for t in 0 1 2 3 4 5 6 7 8 9; do
    echo "normal --tasks $t --episodes 0-19,50-139 --dart 0.1 --seed $t --out data/v6_normal|normal_T$t"
  done
  k=0
  for g in "1:0,1:7,4:3,8:0,8:6" "1:2,1:8,4:5,8:1,8:7" "1:3,1:9,4:6,8:2,8:9" "1:4,4:0,4:7,8:3" "1:5,4:1,4:8,8:4" "1:6,4:2,4:9,8:5"; do
    echo "switch --pairs $g --episodes 0-19,50-69 --dart 0.1 --seed $((100+k)) --out data/v6_switch|switch_$k"
    k=$((k+1))
  done
} | xargs -P "$MAXP" -I{} bash -c 'job="{}"; args="${job%|*}"; name="${job#*|}"; .venv/bin/python collect_v6.py $args > outputs/v6/$name.log 2>&1'
echo "=== $(date +%H:%M) 수집 끝"
