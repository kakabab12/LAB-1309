#!/usr/bin/env bash
# Round 2, v11 part relaunched (10-11 04:10): there are 500 factory demos per stage, not 900.
# Round 2 towards 100 % (10-11 03:45). Both models were still improving when stopped (L1 falling over the last 10k):
# - v11 = v10 + all 500 bin factory-mix demos per stage (p0-p4) with v9's 1000 clutter demos, continued from v10 (50k) for 20k
#   -> runs/s*_v11_n1500/ckpt_070000 -> the six table conditions -> results/clutter/v11_n1500
#   (v10 with 300 factory demos: factory 44 -> 68 %, factory_max 24 -> 48 %)
# - integrated 120k = int80 continued for 40k more (seed 2) -> runs/s*_int120_n1800/ckpt_120000 -> 5 conditions x 6
#   combinations -> results/clutter/int120_n1800_<obj>_<layout>   (40k -> 80k: bin/orig plain 70 -> 88 %, factory 36 -> 74 %)
# v11 trainings go first (shorter), the GPU limits are shared with the orchestrator.
set -u
cd "$(dirname "$0")"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> logs/queue.log; }
n_train() { pgrep -fc "^.venv/bin/python train_act.py" || true; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
slot() { while [ "$(n_train)" -ge 4 ] || [ "$(gpu_free)" -lt 3000 ]; do sleep 10; done; }
say "v11 (fixed: 1000 + 500 factory demos per stage, the first launch asked for 1900 and stopped)"
v11=() int=()  # pids (the evaluation subshells poll them: wait only works for the parent shell's children)
for s in 1 2 3; do
  run=s${s}_v11_n1500
  [ -f runs/$run/ckpt_070000/model.safetensors ] && continue
  slot; say "start train $run (from s${s}_v10_n1300/ckpt_050000)"
  nice -n 10 .venv/bin/python train_act.py --stage $s --episodes 1500 --steps 20000 --save-every 10000 \
      --data data/v9_stage$s.jpk.npz $(for p in 0 1 2 3 4; do echo -n "data/i_bin_orig_s${s}_p$p.jpk.npz "; done) \
      --init runs/s${s}_v10_n1300/ckpt_050000 --step0 50000 --seed 2 --out runs/$run >> logs/train_$run.log 2>&1 &
  v11+=($!); sleep 90
done
# v11 evaluation as soon as its three stages are done
( for pid in "${v11[@]}"; do while kill -0 $pid 2>/dev/null; do sleep 60; done; done
  C="runs/s1_v11_n1500/ckpt_070000 runs/s2_v11_n1500/ckpt_070000 runs/s3_v11_n1500/ckpt_070000"
  for s in 1 2 3; do [ -f runs/s${s}_v11_n1500/ckpt_070000/model.safetensors ] || { say "round2: s${s}_v11_n1500 not trained"; exit 1; }; done
  ev() { CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 .venv/bin/python clutter_eval.py --ckpt $C --tag v11_n1500 \
           --conds "$@" --factory-env --mode chained --trials 50 --gifs 2 > logs/clutter2_v11_n1500_$1.log 2>&1; }
  ev none all & ev tight factory & ev factory_vis & ev factory_max &
  wait
  say "v11: done eval $(for c in none all tight factory factory_vis factory_max; do f=results/clutter/v11_n1500/$c.json
    [ -f $f ] && python3 -c "import json; print('$c', round(100 * json.load(open('$f'))['chained']['cumulative_success'][2]), end='; ')"; done)"
) &
wait
