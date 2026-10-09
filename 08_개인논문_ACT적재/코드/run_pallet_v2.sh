#!/usr/bin/env bash
# Pallet v2: same 2x2x2 task with a wider column gap and the grid moved towards the robot (layout from the
# PALLET_X0 / PALLET_Y0 / PALLET_GAP_Y environment), because in v1 the policies placed bins ~5 mm towards the
# neighbouring column (8 mm gap -> collisions) and the far corner slot was at the arm's reach limit (expert error 9 mm).
# 100 demos per slot -> pal2_s1..8 (30k steps) -> 40 sequences with re-planning every 25 steps.
set -u
cd "$(dirname "$0")"
PY="nice -n 10 .venv/bin/python"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> logs/queue.log; }
n_train() { pgrep -fc "^.venv/bin/python train_act.py" || true; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
mem_avail_gb() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
say "pallet v2 start (x0 $PALLET_X0, y0 $PALLET_Y0, column gap $PALLET_GAP_Y m): 8 x 100 demos"
gen() {  # 0-based slot
  local k=$1 name=pallet2_s$(($1 + 1))
  [ -f data/$name.jpk.npz ] && return 0
  [ -f data/${name}_gen.json ] || nice -n 15 .venv/bin/python gen_pallet.py --slot $k --episodes 100 --seed 61000 --name $name \
      --out data > logs/gen_$name.log 2>&1
  nice -n 15 .venv/bin/python convert_jpeg.py data/$name.hdf5 >> logs/gen_$name.log 2>&1 && rm -f data/$name.hdf5
}
for k in 0 1 2 3 4 5 6 7; do gen $k & sleep 15; done
wait
for k in 1 2 3 4 5 6 7 8; do [ -f data/pallet2_s$k.jpk.npz ] || { say "pallet v2: data/pallet2_s$k.jpk.npz MISSING"; exit 1; }; done
say "pallet v2: demos ready ($(for k in 1 2 3 4 5 6 7 8; do python3 -c "import json;s=json.load(open('data/pallet2_s${k}_gen.json'))['summary'];print(f\"s$k {s['kept']}/{s['tried']} {s['mean_xy_err_mm']:.1f}mm\",end=' ')"; done))"
pids=()
for k in 1 2 3 4 5 6 7 8; do
  run=pal2_s$k
  [ -f runs/$run/ckpt_030000/model.safetensors ] && continue
  while [ "$(n_train)" -ge 5 ] || [ "$(gpu_free)" -lt 2600 ] || [ "$(mem_avail_gb)" -lt 6 ]; do sleep 30; done
  say "train $run (pallet v2 slot $k, 100 demos)"
  $PY train_act.py --stage $k --episodes 100 --steps 30000 --save-every 10000 --data data/pallet2_s$k.jpk.npz \
      --out runs/$run >> logs/train_$run.log 2>&1 &
  pids+=($!)
  sleep 90
done
[ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
for k in 1 2 3 4 5 6 7 8; do [ -f runs/pal2_s$k/ckpt_030000/model.safetensors ] || { say "pallet v2: pal2_s$k has no final checkpoint"; exit 1; }; done
say "pallet v2 policies trained; evaluating 40 sequences with re-planning every 25 steps"
for p in 0 1; do
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 $PY pallet_eval.py --runs pal2_s1 pal2_s2 pal2_s3 pal2_s4 pal2_s5 pal2_s6 pal2_s7 pal2_s8 \
      --k 25 --start $((p * 20)) --trials 20 --gifs $([ $p = 0 ] && echo 2 || echo 0) --out results/eval/pallet2_k25_p$p \
      > logs/eval_pallet2_k25_p$p.log 2>&1 &
done
wait
say "pallet v2 eval done: $(grep -h '\[pallet\]' logs/eval_pallet2_k25_p*.log | tr '\n' ' ')"
