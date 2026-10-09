#!/usr/bin/env bash
# Pallet slot 7 (0-based 6: upper layer, far row, scale-side column, 27 cm from the base) fails ~half the time:
# the bin lands 9-11 mm too far and tips off the 2 mm rim of the bin below. Fix attempt: 200 more demonstrations
# for this slot (300 total, new seeds), retrain as pal_s7b, check the slot alone, then the full 40-sequence eval.
# Launch:  systemd-run --user --unit=act-palfix -p MemoryMax=9G -p Nice=10 -p WorkingDirectory=$PWD \
#            --setenv=MUJOCO_GL=egl --setenv=ACT_DESIGN=v4 --setenv=PALLET_VIA=far --setenv=MALLOC_MMAP_THRESHOLD_=1048576 \
#            /bin/bash $PWD/run_pallet_fix.sh
set -u
cd "$(dirname "$0")"
PY="nice -n 10 .venv/bin/python"
LOG=logs/queue.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
n_train() { pgrep -fc "^.venv/bin/python train_act.py" || true; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
mem_avail_gb() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
say "pallet fix: 200 more demos for slot 7 (4 x 50, seeds 41000-44000)"
parts=()
for p in 0 1 2 3; do
  name=pallet_s7_m$p
  parts+=(data/$name.jpk.npz)
  [ -f data/$name.jpk.npz ] && continue
  ( [ -f data/${name}_gen.json ] || nice -n 15 .venv/bin/python gen_pallet.py --slot 6 --episodes 50 --seed $((41000 + 1000 * p)) \
        --name $name --out data > logs/gen_$name.log 2>&1
    nice -n 15 .venv/bin/python convert_jpeg.py data/$name.hdf5 >> logs/gen_$name.log 2>&1 && rm -f data/$name.hdf5 ) &
  sleep 20
done
wait
for f in "${parts[@]}"; do [ -f "$f" ] || { say "pallet fix: $f MISSING"; exit 1; }; done
say "pallet fix: demos ready, training pal_s7b (300 demos)"
while [ "$(n_train)" -ge 5 ] || [ "$(gpu_free)" -lt 2600 ] || [ "$(mem_avail_gb)" -lt 5 ]; do sleep 30; done
[ -f runs/pal_s7b/ckpt_030000/model.safetensors ] || $PY train_act.py --stage 7 --episodes 300 --steps 30000 --save-every 10000 \
    --data data/pallet_s7.jpk.npz "${parts[@]}" --out runs/pal_s7b >> logs/train_pal_s7b.log 2>&1
[ -f runs/pal_s7b/ckpt_030000/model.safetensors ] || { say "pallet fix: pal_s7b has no final checkpoint"; exit 1; }
say "trained pal_s7b; slot check (20) + full pallet eval (40)"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 $PY pallet_slot_check.py --slot 6 --run pal_s7b --trials 20 --noise 0.003 \
    --out results/pallet/slotcheck_s6b_n0.003.json > logs/slotchk_s6b.log 2>&1
say "pallet fix slot check: $(grep '\[slot' logs/slotchk_s6b.log) (pal_s7: $(grep '\[slot' logs/slotchk_s6_0.003.log))"
for p in 0 1; do
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 $PY pallet_eval.py --runs pal_s1 pal_s2 pal_s3 pal_s4 pal_s5 pal_s6 pal_s7b pal_s8 \
      --start $((p * 20)) --trials 20 --gifs $([ $p = 0 ] && echo 2 || echo 0) --out results/eval/pallet_fix_p$p > logs/eval_pallet_fix_p$p.log 2>&1 &
done
wait
say "pallet fix eval done: $(grep -h '\[pallet\]' logs/eval_pallet_fix_p*.log | tr '\n' ' ')"
