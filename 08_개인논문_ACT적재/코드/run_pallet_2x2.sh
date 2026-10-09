#!/usr/bin/env bash
# 2x2 single-layer pallet: the first four slots of pallet v2 (no new training). Slot 1 with the state-noise policy
# (pal2n05_s1, 20/20 alone) and, for comparison, the original pal2_s1. 40 sequences each, re-plan every 25 steps.
set -u
cd "$(dirname "$0")"
PY="nice -n 10 .venv/bin/python"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> logs/queue.log; }
say "pallet 2x2 (one layer) eval: pal2n05_s1 + pal2_s2..s4 and pal2_s1..s4, 40 sequences each"
for v in n05 orig; do
  s1=$([ $v = n05 ] && echo pal2n05_s1 || echo pal2_s1)
  for p in 0 1; do
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 $PY pallet_eval.py --runs $s1 pal2_s2 pal2_s3 pal2_s4 --k 25 --start $((p * 20)) \
        --trials 20 --gifs $([ $p = 0 ] && echo 2 || echo 0) --out results/eval/pallet2x2_${v}_p$p > logs/eval_pallet2x2_${v}_p$p.log 2>&1 &
  done
done
wait
say "pallet 2x2 eval: $(for v in n05 orig; do echo -n "$v: $(grep -h '\[pallet\]' logs/eval_pallet2x2_${v}_p*.log | tr '\n' ' ') "; done)"
