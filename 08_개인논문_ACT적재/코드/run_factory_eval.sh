#!/usr/bin/env bash
# Factory-cell conditions (factory / factory_vis / factory_max, clutter_eval.py) for the bin models of the clutter test:
# v5_n100 (final, no randomisation), v6_n1000 (randomised looks), v9_n1000 (+ table clutter). Starts after
# run_clutter.sh has finished its evaluations (the CPU is shared), one process per model.
set -u
cd "$(dirname "$0")"
PY="nice -n 12 .venv/bin/python"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> logs/queue.log; }
while ! grep -q "clutter test finished" logs/queue.log && systemctl --user is-active --quiet act-clutter; do sleep 120; done
say "factory eval start: v5_n100, v6_n1000, v9_n1000 on factory / factory_vis / factory_max"
fe() {  # tag s1 s2 s3
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 $PY clutter_eval.py --ckpt $2 $3 $4 --tag $1 --conds factory factory_vis factory_max \
      --trials 50 --gifs 2 > logs/clutter_$1_factory.log 2>&1
  say "factory eval $1: $(grep chained logs/clutter_$1_factory.log | tr '\n' ' ')"
}
fe v5_n100 runs/s1_v4_n100/ckpt_030000 runs/s2_v4_n100/ckpt_030000 runs/s3_v5_n100/ckpt_030000 &
fe v6_n1000 runs/s1_v6_n1000/ckpt_030000 runs/s2_v6_n1000/ckpt_030000 runs/s3_v6_n1000/ckpt_030000 &
[ -f runs/s3_v9_n1000/ckpt_030000/model.safetensors ] && fe v9_n1000 runs/s1_v9_n1000/ckpt_030000 runs/s2_v9_n1000/ckpt_030000 runs/s3_v9_n1000/ckpt_030000 &
wait
say "factory eval finished"
