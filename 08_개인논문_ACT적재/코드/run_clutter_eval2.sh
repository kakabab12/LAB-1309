#!/usr/bin/env bash
# Clutter / factory evaluation of the bin models, chained only (50 sequences), so that it finishes by the weekend
# (the first run_clutter.sh evaluation did per-stage + chained on 9 conditions: ~3 h per condition on the shared CPU).
# Conditions: none, all (8 objects, look-alike, passing object), tight (5 mm from the arm's path), factory,
# factory_vis, factory_max. v5_n100 already has none / decoy / all (both protocols) from run_clutter.sh.
set -u
cd "$(dirname "$0")"
PY="nice -n 10 .venv/bin/python"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> logs/queue.log; }
ev() {  # tag ckpt-prefix-s1 s2 s3, conds...
  local tag=$1 c1=$2 c2=$3 c3=$4; shift 4
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 $PY clutter_eval.py --ckpt $c1 $c2 $c3 --tag $tag --conds "$@" --mode chained \
      --trials 50 --gifs 2 --factory-env > logs/clutter2_${tag}_$1.log 2>&1
}
V5="runs/s1_v4_n100/ckpt_030000 runs/s2_v4_n100/ckpt_030000 runs/s3_v5_n100/ckpt_030000"
V6="runs/s1_v6_n1000/ckpt_030000 runs/s2_v6_n1000/ckpt_030000 runs/s3_v6_n1000/ckpt_030000"
V9="runs/s1_v9_n1000/ckpt_030000 runs/s2_v9_n1000/ckpt_030000 runs/s3_v9_n1000/ckpt_030000"
say "clutter eval (chained only): v5_n100, v6_n1000 now, v9_n1000 when trained"
ev v5_n100 $V5 tight factory & ev v5_n100 $V5 factory_vis factory_max &
ev v6_n1000 $V6 none all tight & ev v6_n1000 $V6 factory factory_vis factory_max &
wait
say "clutter eval: v5 / v6 done: $(for t in v5_n100 v6_n1000; do for c in none all tight factory factory_vis factory_max; do f=results/clutter/$t/$c.json; [ -f $f ] && python3 -c "import json; d=json.load(open('$f')); print('$t $c', round(100*d['chained']['cumulative_success'][2]), end='; ')"; done; done)"
while [ ! -f runs/s3_v9_n1000/ckpt_030000/model.safetensors ] || [ ! -f runs/s2_v9_n1000/ckpt_030000/model.safetensors ] \
      || [ ! -f runs/s1_v9_n1000/ckpt_030000/model.safetensors ]; do sleep 120; done
ev v9_n1000 $V9 none all tight & ev v9_n1000 $V9 factory factory_vis factory_max &
wait
say "clutter eval: v9 done: $(for c in none all tight factory factory_vis factory_max; do f=results/clutter/v9_n1000/$c.json; [ -f $f ] && python3 -c "import json; d=json.load(open('$f')); print('$c', round(100*d['chained']['cumulative_success'][2]), end='; ')"; done)"
