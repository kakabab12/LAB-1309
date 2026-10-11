#!/usr/bin/env bash
# Factory cell + wide looks (user, 10-11 16:20: "공장형 + 외관 변화 이거만 좀 높여봐"). v11 (v9 + 500 factory-mix demos
# per stage) reaches factory 82 % but factory_vis 58 % / factory_max 46 %. Failures by drawn look (v11, 50 each):
# plain table 70 / 61 %, patterned table (stripes, checker, blotches) 25-56 %; light strength and tint make no clear
# difference. In the factory-mix demos only half the episodes had the wide (v7) looks and 60 % were factory scenes,
# so factory + patterned table was ~15 % of them. Same remedy as v9 -> v10 -> v11: 300 more bin demos per stage, all
# factory scenes (8-16 parts, 5-20 mm) with the wide looks (same generator, new seeds 56000+), continue v11 (70k) for
# 20k on v9's 1000 + 500 factory-mix + these 300 -> runs/s*_v12_n1800/ckpt_090000 -> the six table conditions
# (chained 50, same seeds and criteria) -> results/clutter/v12_n1800.
set -u
cd "$(dirname "$0")"
export ACT_OBJECT=bin ACT_LAYOUT=orig ACT_DESIGN=v5
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> logs/queue.log; }
n_train() { pgrep -fc "^.venv/bin/python train_act.py" || true; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
mem_free() { awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo; }
ck() { [ -f "runs/$1/ckpt_090000/model.safetensors" ]; }
say "v12 (v11 + 300 factory demos with wide looks per stage): demos, +20k from v11, then clutter/factory eval"
gen() {  # stage pack
  local name=i_bin_orig_s$1_f$2
  [ -f data/$name.jpk.npz ] && return 0
  nice -n 15 .venv/bin/python gen_data.py --stage $1 --episodes 100 --seed $((56000 + 10 * $2)) --name $name --out data \
      --dr --factory --factory-mix 0 0 --look2 1 > logs/gen_$name.log 2>&1 \
    && nice -n 15 .venv/bin/python convert_jpeg.py data/$name.hdf5 >> logs/gen_$name.log 2>&1 && rm -f "data/$name.hdf5"
}
for s in 1 2 3; do for p in 0 1 2; do
  while [ "$(gpu_free)" -lt 1500 ]; do sleep 10; done
  gen $s $p & sleep 10
done; done
wait
for s in 1 2 3; do for p in 0 1 2; do
  [ -f data/i_bin_orig_s${s}_f$p.jpk.npz ] || { say "v12: data/i_bin_orig_s${s}_f$p.jpk.npz MISSING"; exit 1; }
done; done
say "v12: demos ready"
pids=()
for s in 1 2 3; do
  run=s${s}_v12_n1800
  ck $run && continue
  while [ "$(n_train)" -ge 4 ] || [ "$(gpu_free)" -lt 3000 ] || [ "$(mem_free)" -lt 9000 ]; do sleep 10; done
  say "start train $run (from s${s}_v11_n1500/ckpt_070000)"
  nice -n 10 .venv/bin/python train_act.py --stage $s --episodes 1800 --steps 20000 --save-every 10000 \
      --data data/v9_stage$s.jpk.npz $(for p in 0 1 2 3 4; do echo -n "data/i_bin_orig_s${s}_p$p.jpk.npz "; done) \
      $(for p in 0 1 2; do echo -n "data/i_bin_orig_s${s}_f$p.jpk.npz "; done) \
      --init runs/s${s}_v11_n1500/ckpt_070000 --step0 70000 --seed 3 --out runs/$run >> logs/train_$run.log 2>&1 &
  pids+=($!)
  sleep 120
done
[ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
for s in 1 2 3; do ck s${s}_v12_n1800 || { say "v12: s${s}_v12_n1800 not trained"; exit 1; }; done
say "v12: trained"
C="runs/s1_v12_n1800/ckpt_090000 runs/s2_v12_n1800/ckpt_090000 runs/s3_v12_n1800/ckpt_090000"
ev() { CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 .venv/bin/python clutter_eval.py --ckpt $C --tag v12_n1800 \
         --conds "$@" --factory-env --mode chained --trials 50 --gifs 2 > logs/clutter2_v12_n1800_$1.log 2>&1; }
for c in factory_vis factory_max factory none all tight; do
  while [ "$(gpu_free)" -lt 1500 ]; do sleep 10; done
  ev $c & sleep 20
done
wait
say "v12: done eval $(for c in none all tight factory factory_vis factory_max; do f=results/clutter/v12_n1800/$c.json
  [ -f $f ] && python3 -c "import json; print('$c', round(100 * json.load(open('$f'))['chained']['cumulative_success'][2]), end='; ')"; done)"
