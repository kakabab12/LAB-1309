#!/usr/bin/env bash
# run_v6c.sh 가 끝나면 라운드를 이어서 돌린다: v6c → v6d → v6e (10/2)
cd "$(dirname "$0")"
until grep -qE "^=== [0-9/]+ [0-9:]+ 끝$" outputs/v6/run_v6c.log; do sleep 120; done
./run_round.sh v6c v6d v6clat v6ca outputs/a2c2_v6c > outputs/v6/round_v6d.log 2>&1
grep -q "라운드 끝" outputs/v6/round_v6d.log || exit 1
./run_round.sh v6d v6e v6dlat v6da outputs/a2c2_v6d > outputs/v6/round_v6e.log 2>&1
