#!/usr/bin/env bash
# Push towards 100 % (user, 10-10 19:10). Two things that need no GPU:
# 1) re-planning every 25 steps instead of executing the whole 100-step chunk (helped the pallet: 57.5 -> 70 %):
#    v9 on the clutter / factory conditions, same seeds and criteria -> results/clutter/v9k25_n1000
# 2) more bin factory-cell demonstrations (2 more packs of 100 per stage, same generator and settings as the
#    integrated packs, new seeds) in case v10 (300 factory demos) is not enough -> data/i_bin_orig_s*_p3/p4
set -u
cd "$(dirname "$0")"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> logs/queue.log; }
say "push100: v9 with re-planning every 25 steps (clutter/factory) + 600 more bin factory demos"
V9="runs/s1_v9_n1000/ckpt_030000 runs/s2_v9_n1000/ckpt_030000 runs/s3_v9_n1000/ckpt_030000"
ev() { CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 .venv/bin/python clutter_eval.py --ckpt $V9 --tag v9k25_n1000 \
         --n-action-steps 25 --conds "$@" --factory-env --mode chained --trials 50 --gifs 2 \
         > logs/clutter2_v9k25_$1.log 2>&1; }
( ev factory all; ev factory_vis ) &
( ev tight factory_max; ev none ) &
gen() {  # stage pack
  local name=i_bin_orig_s$1_p$2
  [ -f data/$name.jpk.npz ] && return 0
  ACT_OBJECT=bin ACT_LAYOUT=orig ACT_DESIGN=v5 nice -n 15 .venv/bin/python gen_data.py --stage $1 --episodes 100 \
      --seed $((51000 + 10 * $2)) --name $name --out data --dr --factory > logs/gen_$name.log 2>&1 \
    && nice -n 15 .venv/bin/python convert_jpeg.py data/$name.hdf5 >> logs/gen_$name.log 2>&1 && rm -f "data/$name.hdf5"
}
for s in 1 2 3; do for p in 3 4; do gen $s $p & sleep 5; done; done
wait
say "push100: done eval v9k25 $(for c in none all tight factory factory_vis factory_max; do f=results/clutter/v9k25_n1000/$c.json
  [ -f $f ] && python3 -c "import json; print('$c', round(100 * json.load(open('$f'))['chained']['cumulative_success'][2]), end='; ')"; done)| demos $(ls data/i_bin_orig_s?_p[34].jpk.npz 2>/dev/null | wc -l)/6 packs"
