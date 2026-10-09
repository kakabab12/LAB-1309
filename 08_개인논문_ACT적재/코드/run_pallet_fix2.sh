#!/usr/bin/env bash
# After pal_s7b (slot 7, 300 demos) is trained by act-palfix, stop that queue before its k=100 evaluation and evaluate
# pal_s7b with re-planning every 25 steps instead (the setting that worked best): slot check (20) + full pallet (2 x 20).
set -u
cd "$(dirname "$0")"
PY="nice -n 10 .venv/bin/python"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> logs/queue.log; }
until [ -f runs/pal_s7b/ckpt_030000/model.safetensors ]; do
  systemctl --user is-active --quiet act-palfix || { say "pallet fix2: act-palfix ended without pal_s7b"; exit 1; }
  sleep 60
done
sleep 30
systemctl --user stop act-palfix
say "trained pal_s7b (slot 7, 300 demos); act-palfix stopped before its k=100 eval -> k=25 slot check + full eval"
while systemctl --user is-active --quiet act-palk25; do sleep 60; done
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 $PY pallet_slot_check.py --slot 6 --run pal_s7b --trials 20 --noise 0.003 --k 25 \
    --out results/pallet/slotcheck_s6b_k25.json > logs/slotchk_s6b_k25.log 2>&1
say "slot 7 check pal_s7b k=25: $(grep '\[slot' logs/slotchk_s6b_k25.log) (pal_s7 k=25: $(grep '\[slot' logs/slotchk_s6_k.log))"
for p in 0 1; do
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 $PY pallet_eval.py --runs pal_s1 pal_s2 pal_s3 pal_s4 pal_s5 pal_s6 pal_s7b pal_s8 \
      --k 25 --start $((p * 20)) --trials 20 --gifs $([ $p = 0 ] && echo 2 || echo 0) --out results/eval/pallet_fixk25_p$p \
      > logs/eval_pallet_fixk25_p$p.log 2>&1 &
done
wait
say "pallet eval pal_s7b + k=25 done: $(grep -h '\[pallet\]' logs/eval_pallet_fixk25_p*.log | tr '\n' ' ')"
