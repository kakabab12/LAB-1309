#!/usr/bin/env bash
# Full 2x2x2 pallet evaluation with re-planning every 25 steps for all 8 slot policies (pal_s1..8, unchanged
# weights): in the isolated slot-7 test, re-planning every 25 steps instead of executing whole 100-step chunks
# removed the far-row overshoot. Starts after the first pallet evaluation and the slot checks have finished.
set -u
cd "$(dirname "$0")"
PY="nice -n 10 .venv/bin/python"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> logs/queue.log; }
while systemctl --user is-active --quiet act-paleval0 || systemctl --user is-active --quiet act-paleval1 \
      || systemctl --user is-active --quiet act-slotchk-k || systemctl --user is-active --quiet act-slotchk-te; do sleep 60; done
say "pallet eval (k=100) done: $(grep -h '\[pallet\]' logs/eval_pallet_p0.log logs/eval_pallet_p1.log | tr '\n' ' ')"
say "slot 7 checks: k=100 $(grep '\[slot' logs/slotchk_s6_0.003.log) | k=25 $(grep '\[slot' logs/slotchk_s6_k.log) | TE $(grep '\[slot' logs/slotchk_s6_te.log)"
say "pallet eval with re-planning every 25 steps (all slots, 2 x 20 sequences)"
for p in 0 1; do
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 $PY pallet_eval.py --runs pal_s1 pal_s2 pal_s3 pal_s4 pal_s5 pal_s6 pal_s7 pal_s8 \
      --k 25 --start $((p * 20)) --trials 20 --gifs $([ $p = 0 ] && echo 2 || echo 0) --out results/eval/pallet_k25_p$p \
      > logs/eval_pallet_k25_p$p.log 2>&1 &
done
wait
say "pallet eval k=25 done: $(grep -h '\[pallet\]' logs/eval_pallet_k25_p*.log | tr '\n' ' ')"
