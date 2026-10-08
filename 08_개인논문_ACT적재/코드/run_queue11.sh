#!/usr/bin/env bash
# Queue v11: v7 = final design + wider domain randomisation (v7 data), 1000 demos, no CLAHE
# (v6 without CLAHE >= v6c with CLAHE on all 20 robustness conditions: better on 13, tied on 7; mean 96.4 vs 90.8%).
# Jumps ahead of the remaining lower-priority jobs: allows one extra GPU slot (MAXT=5) while memory permits.
# Then: standard evaluation (GPU) and all 20 robustness conditions (CPU, 3 processes).
# Launch:  systemd-run --user --unit=act-queue11 -p MemoryMax=10G -p Nice=10 -p WorkingDirectory=$PWD \
#            --setenv=MUJOCO_GL=egl --setenv=MALLOC_MMAP_THRESHOLD_=1048576 /bin/bash $PWD/run_queue11.sh
set -u
cd "$(dirname "$0")"
PY="nice -n 10 .venv/bin/python"
MAXT=${MAXT:-5}
LOG=logs/queue.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
ck() { echo "runs/$1/ckpt_030000"; }
done_run() { [ -f "$(ck $1)/model.safetensors" ]; }
n_train() { pgrep -fc "^.venv/bin/python train_act.py" || true; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
mem_avail_gb() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
say "queue v11 start (v7: wider randomisation, no CLAHE, 1000 demos)"
pids=()
for s in 1 2 3; do
  run=s${s}_v7_n1000
  done_run $run && continue
  until [ -f data/v7_stage$s.jpk.npz ]; do sleep 60; done
  while [ "$(n_train)" -ge $MAXT ] || [ "$(gpu_free)" -lt 2600 ] || [ "$(mem_avail_gb)" -lt 5 ]; do sleep 30; done
  say "train $run (1000 demos, v7 data, no CLAHE)"
  $PY train_act.py --stage $s --episodes 1000 --steps 30000 --save-every 10000 --data data/v7_stage$s.jpk.npz \
      --out runs/$run >> logs/train_$run.log 2>&1 &
  pids+=($!)
  sleep 90
done
[ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
for s in 1 2 3; do done_run s${s}_v7_n1000 || { say "v7c: s${s}_v7_n1000 missing"; exit 1; }; done
C="$(ck s1_v7_n1000) $(ck s2_v7_n1000) $(ck s3_v7_n1000)"
say "eval v7_n1000 + robustness (20 conditions)"
[ -f results/eval/v7_n1000/results.json ] || $PY eval_act.py --ckpt $C --trials 50 --out results/eval/v7_n1000 --gifs 3 > logs/eval_v7_n1000.log 2>&1 &
for part in "base dark bright light_side table_gray table_dark very_dark" \
            "distractor cam_top cam_wrist pos_out yaw_out placed_out very_bright" \
            "mass_100g mass_200g delay_67ms delay_133ms warm_light table_checker"; do
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 $PY robust_eval.py --ckpt $C --tag v7_n1000 --trials 20 --conds $part \
      > logs/robust_v7_n1000_$(echo $part | cut -c1-4).log 2>&1 &
done
wait
say "eval v7_n1000 done: $(grep '\[chained\]' logs/eval_v7_n1000.log)"
say "robust v7_n1000 done"
