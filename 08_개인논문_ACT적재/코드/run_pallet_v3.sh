#!/usr/bin/env bash
# Pallet v3 = pallet v2 layout and demonstrations, every slot policy trained with state-input noise 0.05 rad
# (train_act.py --state-noise): the v2 policies drifted 10-20 mm towards +y; with the noise slot 1 went 18/20 -> 20/20
# and slot 5 2/20 -> 9/20 alone. Slot 1 = pal2n05_s1; slot 5 = the better of pal2n05_s5 / pal2n10_s5 (0.1 rad, alone
# check); slots 2-4 and 6-8 retrained here. Then 40 sequences of all 8 slots, re-planning every 25 steps.
set -u
cd "$(dirname "$0")"
PY="nice -n 10 .venv/bin/python"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> logs/queue.log; }
n_train() { pgrep -fc "^.venv/bin/python train_act.py" || true; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
mem_avail_gb() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
say "pallet v3 start: slots 2-4, 6-8 with state noise 0.05 rad (slot 1 = pal2n05_s1, slot 5 = best of 0.05 / 0.1)"
pids=()
for k in 2 3 4 6 7 8; do
  run=pal2n05_s$k
  [ -f runs/$run/ckpt_030000/model.safetensors ] && continue
  while [ "$(n_train)" -ge 6 ] || [ "$(gpu_free)" -lt 2000 ] || [ "$(mem_avail_gb)" -lt 5 ]; do sleep 30; done
  say "train $run (pallet v3, state noise 0.05)"
  $PY train_act.py --stage $k --episodes 100 --steps 30000 --save-every 10000 --data data/pallet2_s$k.jpk.npz \
      --state-noise 0.05 --out runs/$run >> logs/train_$run.log 2>&1 &
  pids+=($!)
  sleep 60
done
[ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
while [ ! -f results/pallet/slotcheck_pal2n10_s5.json ]; do sleep 120; done
s5=$(python3 -c "
import json
a = sum(r['success'] for r in json.load(open('results/pallet/slotcheck_pal2n05_s5.json'))['rows'])
b = sum(r['success'] for r in json.load(open('results/pallet/slotcheck_pal2n10_s5.json'))['rows'])
print('pal2n10_s5' if b > a else 'pal2n05_s5')")
say "pallet v3: all slots trained, slot 5 = $s5; evaluating 40 sequences"
for p in 0 1; do
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 $PY pallet_eval.py --runs pal2n05_s1 pal2n05_s2 pal2n05_s3 pal2n05_s4 $s5 \
      pal2n05_s6 pal2n05_s7 pal2n05_s8 --k 25 --start $((p * 20)) --trials 20 --gifs $([ $p = 0 ] && echo 2 || echo 0) \
      --out results/eval/pallet3_k25_p$p > logs/eval_pallet3_k25_p$p.log 2>&1 &
done
wait
say "pallet v3 eval done: $(grep -h '\[pallet\]' logs/eval_pallet3_k25_p*.log | tr '\n' ' ')"
