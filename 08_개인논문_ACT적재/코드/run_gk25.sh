#!/usr/bin/env bash
# ② cup: the rule demos fixed the missed grasps (31 -> 1) but the side stage places ~12 mm off (mean dxy +6 / +10 mm,
# the same +y drift as the bin side stage and the pallet). Re-planning every 25 steps (no training) on both designs,
# same 50 sequences -> results/eval/g_cup_orig_<design>_k25.
set -u
cd "$(dirname "$0")"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> logs/queue.log; }
ev() {  # obj design
  local t=g_$1_orig_$2
  ACT_OBJECT=$1 ACT_LAYOUT=orig ACT_DESIGN=$2 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 .venv/bin/python eval_act.py --mode chained \
      --device cpu --ckpt runs/${t}_s1/ckpt_030000 runs/${t}_s2/ckpt_030000 runs/${t}_s3/ckpt_030000 --trials 50 \
      --gifs 2 --n-action-steps 25 --out results/eval/${t}_k25 > logs/eval_${t}_k25.log 2>&1
  say "done eval ${t}_k25 $(python3 -c "import json; r=json.load(open('results/eval/${t}_k25/results.json')); print('chained', ' -> '.join(f'{100*c:.0f}%' for c in r['chained']['cumulative_success']))" 2>/dev/null)"
}
ev cup cuprule & ev cup cupnaive &
wait
