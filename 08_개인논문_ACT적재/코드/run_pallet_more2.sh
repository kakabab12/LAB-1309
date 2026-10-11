#!/usr/bin/env bash
# Pallet 2x2x2, second push (10-11 15:10): after the first push the 8-stack is 82.5 % (33/40), slots
# 100/95/100/100/98/90/98/100 -> 100 more demos for slot 2 (from pal2n05_s2) and slot 6 (on top of its 200, from
# pal3n05_s6@50000) -> pal4n05_s{2,6} -> full eval pallet6. The first push, for reference:
# Pallet 2x2x2, last push (10-11 07:30). With the slot-5 fix the full 8-stack is 65 % (40 sequences); the weak slots
# are 1 (92 %, +7 mm y bias), 4 (92 %, +9 mm), 6-8 (85-88 %, occasional large misses, y std 10-21 mm). Same remedy that
# took the cup from 46 to 88 %: 100 more demonstrations per weak slot (position-first IK expert, new seeds) and state
# noise 0.05, continued from the pal2n05 policies for 20k steps -> runs/pal3n05_s{1,4,6,7,8}/ckpt_050000, then the
# same full evaluation (2 x 20 sequences, k 25) with slot 2, 3 = pal2n05 and slot 5 = pal3n05_s5
# -> results/eval/pallet6_k25_p0/1.
set -u
cd "$(dirname "$0")"
export PALLET_X0=0.20 PALLET_Y0=0.017 PALLET_GAP_Y=0.014 PALLET_VIA=far ACT_DESIGN=v4
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> logs/queue.log; }
n_train() { pgrep -fc "^.venv/bin/python train_act.py" || true; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
SLOTS="2 6"
say "pallet more2: +100 demos for slots $SLOTS, then full 2x2x2 eval"
for k in $SLOTS; do
  [ -f data/pallet4_s${k}_x.jpk.npz ] && continue
  for p in 0 1 2 3; do
    [ -f data/pallet4_s${k}_x${p}_gen.json ] || PALLET_IK_POS=1 nice -n 15 .venv/bin/python gen_pallet.py --slot $((k - 1)) \
        --episodes 25 --seed $((65000 + p)) --name pallet4_s${k}_x$p --out data > logs/gen_pallet4_s${k}_x$p.log 2>&1 &
    sleep 5
  done
done
wait
for k in $SLOTS; do
  [ -f data/pallet4_s${k}_x.jpk.npz ] && continue
  nice -n 15 .venv/bin/python convert_jpeg.py data/pallet4_s${k}_x0.hdf5 data/pallet4_s${k}_x1.hdf5 data/pallet4_s${k}_x2.hdf5 data/pallet4_s${k}_x3.hdf5 \
      --out data/pallet4_s${k}_x.jpk.npz > logs/gen_pallet4_s${k}_x.log 2>&1 \
    && rm -f "data/pallet4_s${k}_x0.hdf5" "data/pallet4_s${k}_x1.hdf5" "data/pallet4_s${k}_x2.hdf5" "data/pallet4_s${k}_x3.hdf5"
  [ -f data/pallet4_s${k}_x.jpk.npz ] || { say "pallet more2: data/pallet4_s${k}_x.jpk.npz MISSING"; exit 1; }
done
say "pallet more2: demos ready"
pids=()
train() {  # slot init-run init-step episodes data...
  local k=$1 init=$2 st=$3 ep=$4; shift 4
  local run=pal4n05_s$k end=$((st + 20000))
  [ -f runs/$run/ckpt_$(printf %06d $end)/model.safetensors ] && return 0
  while [ "$(n_train)" -ge 4 ] || [ "$(gpu_free)" -lt 3000 ]; do sleep 10; done
  say "start train $run (from $init/ckpt_$(printf %06d $st), $ep demos)"
  nice -n 10 .venv/bin/python train_act.py --stage $k --episodes $ep --steps 20000 --save-every 10000 --data "$@" \
      --state-noise 0.05 --init runs/$init/ckpt_$(printf %06d $st) --step0 $st --seed 2 --out runs/$run \
      >> logs/train_$run.log 2>&1 &
  pids+=($!)
  sleep 60
}
train 2 pal2n05_s2 30000 200 data/pallet2_s2.jpk.npz data/pallet4_s2_x.jpk.npz
train 6 pal3n05_s6 50000 300 data/pallet2_s6.jpk.npz data/pallet3_s6_x.jpk.npz data/pallet4_s6_x.jpk.npz
[ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
[ -f runs/pal4n05_s2/ckpt_050000/model.safetensors ] && [ -f runs/pal4n05_s6/ckpt_070000/model.safetensors ] || { say "pallet more2: not trained"; exit 1; }
say "pallet more2: trained -> full 2x2x2 eval"
for p in 0 1; do
  [ -f results/eval/pallet6_k25_p$p/results.json ] && continue
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 .venv/bin/python pallet_eval.py --runs pal3n05_s1@50000 pal4n05_s2@50000 \
      pal2n05_s3 pal3n05_s4@50000 pal3n05_s5 pal4n05_s6@70000 pal3n05_s7@50000 pal3n05_s8@50000 --k 25 \
      --start $((p * 20)) --trials 20 --gifs $([ $p = 0 ] && echo 2 || echo 0) --out results/eval/pallet6_k25_p$p \
      > logs/eval_pallet6_p$p.log 2>&1 &
done
wait
say "pallet more: done eval pallet5 ($(for p in 0 1; do grep -h "full" logs/eval_pallet6_p$p.log | tail -1; done | tr '\n' ' '))"
