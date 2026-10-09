#!/usr/bin/env bash
# Pallet slot 5 (and slot 1) drift test: the v2 slot policies place 10-20 mm towards +y (the direction the bin travels
# from the scale) although the expert is within ~5 mm. Hypothesis: the policy copies its own lagging joint state
# into the action chunk. Retrain with noise on the state input (train_act.py --state-noise) and compare the slot
# alone (pallet_slot_check.py, 20 trials, re-plan every 25 steps).
set -u
cd "$(dirname "$0")"
PY="nice -n 10 .venv/bin/python"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> logs/queue.log; }
say "pallet drift test: slot 5 / slot 1 policies with state-input noise 0.02 / 0.05 rad"
pids=()
for job in "5 02 0.02" "5 05 0.05" "1 05 0.05"; do set -- $job
  run=pal2n$2_s$1
  [ -f runs/$run/ckpt_030000/model.safetensors ] && continue
  $PY train_act.py --stage $1 --episodes 100 --steps 30000 --save-every 10000 --data data/pallet2_s$1.jpk.npz \
      --state-noise $3 --out runs/$run >> logs/train_$run.log 2>&1 &
  pids+=($!); sleep 60
done
[ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
for job in "5 02" "5 05" "1 05"; do set -- $job
  run=pal2n$2_s$1
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 $PY pallet_slot_check.py --slot $(($1 - 1)) --run $run --trials 20 --k 25 \
      --out results/pallet/slotcheck_$run.json > logs/slotcheck_$run.log 2>&1 &
done
wait
say "pallet drift test: $(for r in pal2n02_s5 pal2n05_s5 pal2n05_s1; do echo -n "$r $(grep '^\[slot' logs/slotcheck_$r.log) "; done)(before: slot 5 2/20, slot 1 18/20)"
