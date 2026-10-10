#!/usr/bin/env bash
# ② cup, rule design pushed further. With 100 rule demos the cup policy grasps reliably (missed picks 31 -> 1 vs the
# human-like design) but places 10-12 mm off even when it succeeds (expert 3-5 mm) and drifts at the side stage
# (mean +6 / +10 mm): 46 % chained. Re-planning every 25 steps did not help (42 %). The comparison table stays at
# 100 demos for both designs; this is an extra row: rule design with 300 demos (+200 new, same generator, new seeds)
# and state noise 0.05, continued from the 100-demo policies (+20k steps) -> runs/g_cup_orig_cuprule300_s*/ckpt_050000
# -> chained 50 (same seeds and criteria) -> results/eval/g_cup_orig_cuprule300.
set -u
cd "$(dirname "$0")"
export ACT_OBJECT=cup ACT_LAYOUT=orig ACT_DESIGN=cuprule
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> logs/queue.log; }
n_train() { pgrep -fc "^.venv/bin/python train_act.py" || true; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
ck() { [ -f "runs/$1/ckpt_050000/model.safetensors" ]; }
say "cup300: +200 rule demos per stage, state noise 0.05, continue from the 100-demo cup rule policies"
gen() {  # stage pack
  local name=g_cup_orig_cuprule_s$1_x$2
  [ -f data/$name.jpk.npz ] && return 0
  nice -n 15 .venv/bin/python gen_data.py --stage $1 --episodes 100 --seed $((72000 + 100 * $1 + 10 * $2)) --name $name \
      --out data > logs/gen_$name.log 2>&1 \
    && nice -n 15 .venv/bin/python convert_jpeg.py data/$name.hdf5 >> logs/gen_$name.log 2>&1 && rm -f "data/$name.hdf5"
}
for s in 1 2 3; do for p in 1 2; do gen $s $p & sleep 5; done; done
wait
for s in 1 2 3; do for p in 1 2; do [ -f data/g_cup_orig_cuprule_s${s}_x$p.jpk.npz ] || { say "cup300: demos MISSING"; exit 1; }; done; done
say "cup300: demos ready"
pids=()
for s in 1 2 3; do
  run=g_cup_orig_cuprule300_s$s
  ck $run && continue
  while [ "$(n_train)" -ge 4 ] || [ "$(gpu_free)" -lt 3000 ]; do sleep 10; done
  say "start train $run"
  nice -n 10 .venv/bin/python train_act.py --stage $s --episodes 300 --steps 20000 --save-every 10000 \
      --data data/g_cup_orig_cuprule_s$s.jpk.npz data/g_cup_orig_cuprule_s${s}_x1.jpk.npz \
      data/g_cup_orig_cuprule_s${s}_x2.jpk.npz --state-noise 0.05 --init runs/g_cup_orig_cuprule_s$s/ckpt_030000 \
      --step0 30000 --seed 1 --out runs/$run >> logs/train_$run.log 2>&1 &
  pids+=($!)
  sleep 90
done
[ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
for s in 1 2 3; do ck g_cup_orig_cuprule300_s$s || { say "cup300: g_cup_orig_cuprule300_s$s not trained"; exit 1; }; done
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 .venv/bin/python eval_act.py --mode chained --device cpu \
    --ckpt runs/g_cup_orig_cuprule300_s1/ckpt_050000 runs/g_cup_orig_cuprule300_s2/ckpt_050000 \
    runs/g_cup_orig_cuprule300_s3/ckpt_050000 --trials 50 --gifs 2 --out results/eval/g_cup_orig_cuprule300 \
    > logs/eval_g_cup_orig_cuprule300.log 2>&1
say "cup300: done eval $(python3 -c "import json; r=json.load(open('results/eval/g_cup_orig_cuprule300/results.json')); print('chained', ' -> '.join(f'{100*c:.0f}%' for c in r['chained']['cumulative_success']))")"
