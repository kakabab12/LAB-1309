#!/usr/bin/env bash
# Pallet 2x2x2, last push (10-11 07:30). With the slot-5 fix the full 8-stack is 65 % (40 sequences); the weak slots
# are 1 (92 %, +7 mm y bias), 4 (92 %, +9 mm), 6-8 (85-88 %, occasional large misses, y std 10-21 mm). Same remedy that
# took the cup from 46 to 88 %: 100 more demonstrations per weak slot (position-first IK expert, new seeds) and state
# noise 0.05, continued from the pal2n05 policies for 20k steps -> runs/pal3n05_s{1,4,6,7,8}/ckpt_050000, then the
# same full evaluation (2 x 20 sequences, k 25) with slot 2, 3 = pal2n05 and slot 5 = pal3n05_s5
# -> results/eval/pallet5_k25_p0/1.
set -u
cd "$(dirname "$0")"
export PALLET_X0=0.20 PALLET_Y0=0.017 PALLET_GAP_Y=0.014 PALLET_VIA=far ACT_DESIGN=v4
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> logs/queue.log; }
n_train() { pgrep -fc "^.venv/bin/python train_act.py" || true; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
SLOTS="1 4 6 7 8"
say "pallet more: +100 demos and state noise for slots $SLOTS, then full 2x2x2 eval"
for k in $SLOTS; do
  [ -f data/pallet3_s${k}_x.jpk.npz ] && continue
  for p in 0 1; do
    [ -f data/pallet3_s${k}_x${p}_gen.json ] || PALLET_IK_POS=1 nice -n 15 .venv/bin/python gen_pallet.py --slot $((k - 1)) \
        --episodes 50 --seed $((64000 + p)) --name pallet3_s${k}_x$p --out data > logs/gen_pallet3_s${k}_x$p.log 2>&1 &
    sleep 5
  done
done
wait
for k in $SLOTS; do
  [ -f data/pallet3_s${k}_x.jpk.npz ] && continue
  nice -n 15 .venv/bin/python convert_jpeg.py data/pallet3_s${k}_x0.hdf5 data/pallet3_s${k}_x1.hdf5 \
      --out data/pallet3_s${k}_x.jpk.npz > logs/gen_pallet3_s${k}_x.log 2>&1 \
    && rm -f "data/pallet3_s${k}_x0.hdf5" "data/pallet3_s${k}_x1.hdf5"
  [ -f data/pallet3_s${k}_x.jpk.npz ] || { say "pallet more: data/pallet3_s${k}_x.jpk.npz MISSING"; exit 1; }
done
say "pallet more: demos ready"
pids=()
for k in $SLOTS; do
  run=pal3n05_s$k
  [ -f runs/$run/ckpt_050000/model.safetensors ] && continue
  while [ "$(n_train)" -ge 4 ] || [ "$(gpu_free)" -lt 3000 ]; do sleep 10; done
  say "start train $run (from pal2n05_s$k/ckpt_030000, 200 demos)"
  nice -n 10 .venv/bin/python train_act.py --stage $k --episodes 200 --steps 20000 --save-every 10000 \
      --data data/pallet2_s$k.jpk.npz data/pallet3_s${k}_x.jpk.npz --state-noise 0.05 \
      --init runs/pal2n05_s$k/ckpt_030000 --step0 30000 --seed 1 --out runs/$run >> logs/train_$run.log 2>&1 &
  pids+=($!)
  sleep 90
done
[ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
for k in $SLOTS; do [ -f runs/pal3n05_s$k/ckpt_050000/model.safetensors ] || { say "pallet more: pal3n05_s$k not trained"; exit 1; }; done
say "pallet more: trained -> full 2x2x2 eval"
for p in 0 1; do
  [ -f results/eval/pallet5_k25_p$p/results.json ] && continue
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 .venv/bin/python pallet_eval.py --runs pal3n05_s1@50000 pal2n05_s2 \
      pal2n05_s3 pal3n05_s4@50000 pal3n05_s5 pal3n05_s6@50000 pal3n05_s7@50000 pal3n05_s8@50000 --k 25 \
      --start $((p * 20)) --trials 20 --gifs $([ $p = 0 ] && echo 2 || echo 0) --out results/eval/pallet5_k25_p$p \
      > logs/eval_pallet5_p$p.log 2>&1 &
done
wait
say "pallet more: done eval pallet5 ($(for p in 0 1; do grep -h "full" logs/eval_pallet5_p$p.log | tail -1; done | tr '\n' ' '))"
