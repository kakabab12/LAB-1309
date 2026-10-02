#!/usr/bin/env bash
# run_v6c_post.sh 가 끝나면 라운드를 이어서 돌린다: v6c → v6d → v6e (10/2, 평가는 새 배치 1000번대)
cd "$(dirname "$0")"
until grep -qE "^=== [0-9/]+ [0-9:]+ 끝$" outputs/v6/run_v6c_post.log 2>/dev/null; do sleep 120; done
./run_round.sh v6c v6d v6ch v6cah outputs/a2c2_v6c > outputs/v6/round_v6d.log 2>&1
grep -q "라운드 끝" outputs/v6/round_v6d.log || exit 1
./run_round.sh v6d v6e v6dh v6dah outputs/a2c2_v6d > outputs/v6/round_v6e.log 2>&1
