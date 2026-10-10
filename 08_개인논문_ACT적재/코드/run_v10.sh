#!/usr/bin/env bash
# v10 = v9 + factory-cell demonstrations (bin, original layout). v9 (1000 clutter demos) reached 88 % with clutter but
# only 44 % in the factory cell it never saw; the integrated one-policy-for-3-objects model was weaker still (40k: 36 %).
# Like v6 -> v9 for clutter (58 -> 88 %), put the factory scenes into the demonstrations of the bin policy itself:
# continue v9 from ckpt_030000 for 20k steps on v9's 1000 demos + the 300 bin/orig factory-mix demos already made for
# the integrated model (i_bin_orig_s*_p0-2: 15 % plain / 25 % clutter / 60 % factory, randomized looks).
# -> runs/s{1,2,3}_v10_n1300/ckpt_050000, then chained 50 on the six table conditions -> results/clutter/v10_n1300.
set -u
cd "$(dirname "$0")"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> logs/queue.log; }
n_train() { pgrep -fc "^.venv/bin/python train_act.py" || true; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
ck() { [ -f "runs/$1/ckpt_050000/model.safetensors" ]; }
say "v10 (v9 + bin factory demos): fine-tune s1-s3 from v9 30k (+20k steps, 1300 demos), then clutter/factory eval"
pids=()
for s in 1 2 3; do
  run=s${s}_v10_n1300
  ck $run && continue
  while [ "$(n_train)" -ge 4 ] || [ "$(gpu_free)" -lt 3000 ]; do sleep 10; done
  say "start train $run (from s${s}_v9_n1000/ckpt_030000)"
  nice -n 10 .venv/bin/python train_act.py --stage $s --episodes 1300 --steps 20000 --save-every 10000 \
      --data data/v9_stage$s.jpk.npz data/i_bin_orig_s${s}_p0.jpk.npz data/i_bin_orig_s${s}_p1.jpk.npz \
      data/i_bin_orig_s${s}_p2.jpk.npz --init runs/s${s}_v9_n1000/ckpt_030000 --step0 30000 --seed 1 \
      --out runs/$run >> logs/train_$run.log 2>&1 &
  pids+=($!)
  sleep 90
done
[ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
for s in 1 2 3; do ck s${s}_v10_n1300 || { say "v10: s${s}_v10_n1300 not trained"; exit 1; }; done
say "v10: trained"
C="runs/s1_v10_n1300/ckpt_050000 runs/s2_v10_n1300/ckpt_050000 runs/s3_v10_n1300/ckpt_050000"
ev() { CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 .venv/bin/python clutter_eval.py --ckpt $C --tag v10_n1300 \
         --conds "$@" --factory-env --mode chained --trials 50 --gifs 2 > logs/clutter2_v10_n1300_$1.log 2>&1; }
ev none all & ev tight factory & ev factory_vis & ev factory_max &
wait
say "v10: done eval $(for c in none all tight factory factory_vis factory_max; do f=results/clutter/v10_n1300/$c.json
  [ -f $f ] && python3 -c "import json; print('$c', round(100 * json.load(open('$f'))['chained']['cumulative_success'][2]), end='; ')"; done)"
