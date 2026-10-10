#!/usr/bin/env bash
# Integrated policy, continued training. At 40k steps (int_n1800) the one-policy-per-stage model for 3 objects x 2
# layouts x factory scenes was still learning (L1 0.086-0.095 at 40k and falling, v9 reached 0.075-0.078 at 30k with
# 0.55x the data) and its plain-scene chained success was far below the single-object models (bin 70 / 58, box 64 / 70,
# cup 0 / 24 %). Continue each stage from ckpt_040000 for 40k more steps (same 1800 demos, state noise 0.05, new sampling
# seed, fresh optimizer) -> runs/s{1,2,3}_int80_n1800/ckpt_080000, then the same evaluation (5 conditions x 6 combos,
# chained 50) -> results/clutter/int80_n1800_<obj>_<layout>.
set -u
cd "$(dirname "$0")"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> logs/queue.log; }
n_train() { pgrep -fc "^.venv/bin/python train_act.py" || true; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
ck() { [ -f "runs/$1/ckpt_080000/model.safetensors" ]; }
say "integrated 80k: continue s1-s3 int_n1800 from 40k (+40k steps each), then re-evaluate"
pids=()
for s in 1 2 3; do
  run=s${s}_int80_n1800
  ck $run && continue
  data=$(for o in bin_orig bin_mirror cup_orig cup_mirror box_orig box_mirror; do for p in 0 1 2; do
           echo -n "data/i_${o}_s${s}_p$p.jpk.npz "; done; done)
  while [ "$(n_train)" -ge 4 ] || [ "$(gpu_free)" -lt 3000 ]; do sleep 10; done
  say "start train $run (from s${s}_int_n1800/ckpt_040000)"
  nice -n 10 .venv/bin/python train_act.py --stage $s --episodes 1800 --steps 40000 --save-every 10000 --data $data \
      --state-noise 0.05 --init runs/s${s}_int_n1800/ckpt_040000 --step0 40000 --seed 1 --out runs/$run \
      >> logs/train_$run.log 2>&1 &
  pids+=($!)
  sleep 90
done
[ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
for s in 1 2 3; do ck s${s}_int80_n1800 || { say "integrated 80k: s${s}_int80_n1800 not trained"; exit 1; }; done
say "integrated 80k: trained"
ev() {  # obj layout design
  local tag=int80_n1800_$1_$2
  ACT_OBJECT=$1 ACT_LAYOUT=$2 ACT_DESIGN=$3 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 .venv/bin/python \
      clutter_eval.py --ckpt runs/s1_int80_n1800/ckpt_080000 runs/s2_int80_n1800/ckpt_080000 \
      runs/s3_int80_n1800/ckpt_080000 --tag $tag --conds none all factory factory_vis factory_max --factory-env \
      --mode chained --trials 50 --gifs 2 > logs/eval_$tag.log 2>&1
}
ev bin orig v5 & ev bin mirror v5 & ev cup orig cuprule & ev cup mirror cuprule & ev box orig v5 & ev box mirror v5 &
wait
say "integrated 80k: done eval $(for t in bin_orig bin_mirror cup_orig cup_mirror box_orig box_mirror; do
  echo -n "$t "; for c in none factory; do f=results/clutter/int80_n1800_$t/$c.json
  [ -f $f ] && python3 -c "import json; print(round(100 * json.load(open('$f'))['chained']['cumulative_success'][2]), end='/')"; done
  echo -n "; "; done)(none/factory)"
