#!/usr/bin/env bash
# 2x2 x 2-layer pallet: one ACT policy per slot (8, 100 demos each), after the main queues have launched
# everything; then 8-slot sequence evaluation with scale-verified retry on the CPU (4 x 10 sequences).
# Launch:  systemd-run --user --unit=act-queue-pal -p MemoryMax=8G -p Nice=10 -p WorkingDirectory=$PWD \
#            --setenv=MUJOCO_GL=egl --setenv=ACT_DESIGN=v4 --setenv=MALLOC_MMAP_THRESHOLD_=1048576 /bin/bash $PWD/run_queue_pallet.sh
set -u
cd "$(dirname "$0")"
PY="nice -n 10 .venv/bin/python"
MAXT=${MAXT:-4}
LOG=logs/queue.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
done_run() { [ -f "runs/$1/ckpt_030000/model.safetensors" ]; }
n_train() { pgrep -fc "^.venv/bin/python train_act.py" || true; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
mem_avail_gb() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }

say "pallet queue waiting for queue v10 to launch all its jobs"
until grep -q "queue v10: all trainings launched" $LOG; do sleep 120; done
pids=()
for k in 1 2 3 4 5 6 7 8; do
  run=pal_s$k
  done_run $run && continue
  until [ -f data/pallet_s$k.jpk.npz ]; do sleep 60; done
  while [ "$(n_train)" -ge $MAXT ] || [ "$(gpu_free)" -lt 2600 ] || [ "$(mem_avail_gb)" -lt 5 ]; do sleep 30; done
  say "train $run (pallet slot $k, 100 demos)"
  $PY train_act.py --stage $k --episodes 100 --steps 30000 --save-every 10000 --data data/pallet_s$k.jpk.npz \
      --out runs/$run >> logs/train_$run.log 2>&1 &
  pids+=($!)
  sleep 90
done
[ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
for k in 1 2 3 4 5 6 7 8; do done_run pal_s$k || { say "pallet: pal_s$k has no final checkpoint"; exit 1; }; done
say "pallet policies trained; evaluating 40 sequences (CPU)"
for p in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 $PY pallet_eval.py --runs pal_s1 pal_s2 pal_s3 pal_s4 pal_s5 pal_s6 pal_s7 pal_s8 \
      --start $((p * 10)) --trials 10 --gifs $([ $p = 0 ] && echo 2 || echo 0) --out results/eval/pallet_p$p > logs/eval_pallet_p$p.log 2>&1 &
done
wait
say "pallet eval done: $(grep -h '\[pallet\]' logs/eval_pallet_p*.log | tr '\n' ' ')"
