#!/usr/bin/env bash
# Queue v12: stage-2 policy on v8 demos (keep a gap to the actual bin A), 100 demos, then the final model
# v8_n100 = s1_v4_n100 + s2_v8_n100 + s3_v5_n100: standard eval (GPU, 50), 200-trial eval and 200-sequence
# scale-verified retry eval (CPU, 4 x 50 each).
# Launch:  systemd-run --user --unit=act-queue12 -p MemoryMax=10G -p Nice=10 -p WorkingDirectory=$PWD \
#            --setenv=MUJOCO_GL=egl --setenv=MALLOC_MMAP_THRESHOLD_=1048576 /bin/bash $PWD/run_queue12.sh
set -u
cd "$(dirname "$0")"
PY="nice -n 10 .venv/bin/python"
LOG=logs/queue.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
ck() { echo "runs/$1/ckpt_030000"; }
n_train() { pgrep -fc "^.venv/bin/python train_act.py" || true; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
say "queue v12 start (stage 2 on v8 demos)"
if [ ! -f "$(ck s2_v8_n100)/model.safetensors" ]; then
  until [ -f data/v8_stage2.jpk.npz ]; do sleep 60; done
  while [ "$(n_train)" -ge 5 ] || [ "$(gpu_free)" -lt 2600 ]; do sleep 30; done
  say "train s2_v8_n100 (100 demos, v8)"
  $PY train_act.py --stage 2 --episodes 100 --steps 30000 --save-every 10000 --data data/v8_stage2.jpk.npz \
      --out runs/s2_v8_n100 >> logs/train_s2_v8_n100.log 2>&1
fi
[ -f "$(ck s2_v8_n100)/model.safetensors" ] || { say "s2_v8_n100 missing"; exit 1; }
C="$(ck s1_v4_n100) $(ck s2_v8_n100) $(ck s3_v5_n100)"
say "eval v8_n100 (GPU 50 + CPU 200 + retry 200)"
[ -f results/eval/v8_n100/results.json ] || $PY eval_act.py --ckpt $C --trials 50 --out results/eval/v8_n100 --gifs 3 > logs/eval_v8_n100.log 2>&1 &
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 $PY eval_act.py --ckpt $C --device cpu --start $((k * 50)) --trials 50 --gifs 0 \
      --out results/eval/v8_n100_p$k > logs/eval_v8_n100_p$k.log 2>&1 &
done
wait
$PY merge_eval.py --out results/eval/v8_n100_x200 results/eval/v8_n100_p{0,1,2,3} >> logs/eval_v8_n100.log 2>&1
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 $PY eval_retry.py --ckpt $C --start $((k * 50)) --trials 50 \
      --out results/eval/v8_n100_retry_p$k > logs/retry_v8_n100_p$k.log 2>&1 &
done
wait
say "eval v8_n100 done: $(grep '\[chained\]' logs/eval_v8_n100.log | head -1) | x200: $(tail -n 1 logs/eval_v8_n100.log)"
say "retry v8_n100: $(grep -h '\[retry\]' logs/retry_v8_n100_p*.log | tr '\n' ' ')"
