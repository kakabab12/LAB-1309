#!/usr/bin/env bash
# Experiment queue v5 (user order 10/7: 100 -> 200 -> 500 -> 1000, then DART 1000 and DART 200): in-RAM JPEG packs, memory-aware parallelism (shares the PC with PiPER).
# Launch:  systemd-run --user --unit=act-queue -p MemoryMax=6G -p MemoryHigh=5500M -p Nice=10 -p IOWeight=50 \
#            -p WorkingDirectory=$PWD --setenv=MUJOCO_GL=egl --setenv=MALLOC_MMAP_THRESHOLD_=1048576 \
#            /bin/bash $PWD/run_queue5.sh
set -u
cd "$(dirname "$0")"
PY="nice -n 10 .venv/bin/python"
PYLOW="nice -n 15 .venv/bin/python"
STEPS=${STEPS:-30000}
SAVE=${SAVE:-10000}
TRIALS=${TRIALS:-50}
LOG=logs/queue.log
mkdir -p logs runs results/eval
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
wait_files() { for f in "$@"; do until [ -f "$f" ]; do sleep 30; done; done; }

train_round() {  # name, episodes, max_parallel, data...
  local name=$1 eps=$2 maxpar=$3; shift 3
  local data="$*"
  local pids=()
  for s in 1 2 3; do
    if [ -f "runs/s${s}_${name}/ckpt_$(printf %06d $STEPS)/model.safetensors" ]; then continue; fi
    while [ "$(jobs -rp | grep -c -F -f <(printf '%s\n' "${pids[@]:-none}"))" -ge "$maxpar" ]; do sleep 20; done
    $PY train_act.py --stage $s --episodes $eps --steps $STEPS --save-every $SAVE \
        --data $data --out runs/s${s}_${name} > logs/train_s${s}_${name}.log 2>&1 &
    pids+=($!)
    sleep 60  # stagger start-up memory peaks
  done
  [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
}

evalset() {  # name, step, extra args...
  local name=$1 step=$2; shift 2
  local c=$(printf %06d $step)
  $PY eval_act.py --ckpt runs/s1_${name}/ckpt_$c runs/s2_${name}/ckpt_$c runs/s3_${name}/ckpt_$c \
      --trials $TRIALS "$@" > logs/eval_${name}_${c}${EVAL_TAG:-}.log 2>&1
}

run_round() {  # name, episodes, max_parallel, data...
  local name=$1
  say "train $name"
  train_round "$@"
  say "$name trained; evaluating"
  [ -f results/eval/$name/results.json ] || evalset $name $STEPS --out results/eval/$name --gifs 3
  case $name in n100|n200) ;; *) say "eval $name done"; return ;; esac  # learning-curve evals only for n100/n200
  for mid in $((STEPS / 3)) $((2 * STEPS / 3)); do
    [ -f results/eval/${name}_ckpt$mid/results.json ] || \
      EVAL_TAG=_mid evalset $name $mid --out results/eval/${name}_ckpt$mid --gifs 0 --mode per_stage --trials 30
  done
  say "eval $name done"
}

gen_pack() {  # stem, gen args...
  local stem=$1; shift
  if [ ! -f data/${stem}.jpk.npz ]; then
    [ -f data/${stem}_gen.json ] || $PYLOW gen_data.py --name $stem --out data "$@" > logs/gen_${stem}.log 2>&1
    $PYLOW convert_jpeg.py data/${stem}.hdf5 >> logs/gen_${stem}.log 2>&1
  fi
}

say "queue v5 start (steps=$STEPS save=$SAVE trials=$TRIALS)"

# CPU side, one generator at a time: DART demos first, then 800 more demos per stage
( for s in 1 2 3; do gen_pack dart_stage$s --stage $s --episodes 200 --seed 3000 --dart-sigma 0.005; done
  echo done > logs/dart_data.done
  for s in 1 2 3; do gen_pack stage${s}_more --stage $s --episodes 800 --seed 5000; done
  echo done > logs/more_data.done ) &

variant() {  # name, tag, extra eval args: inference-only variants of a trained model set
  local name=$1 tag=$2; shift 2
  [ -f results/eval/${name}_${tag}/results.json ] || { EVAL_TAG=_$tag evalset $name $STEPS --out results/eval/${name}_${tag} --gifs 2 "$@"; say "eval ${name}_${tag} done"; }
}

run_round n100 100 3 "data/stage{stage}.jpk.npz"
variant n100 k25 --n-action-steps 25
run_round n200 200 3 "data/stage{stage}.jpk.npz"
variant n200 k25 --n-action-steps 25
variant n200 te --temporal-ensemble 0.01
wait_files logs/more_data.done
run_round n500 500 2 "data/stage{stage}.jpk.npz data/stage{stage}_more.jpk.npz"
variant n500 k25 --n-action-steps 25
run_round n1000 1000 1 "data/stage{stage}.jpk.npz data/stage{stage}_more.jpk.npz"
variant n1000 k25 --n-action-steps 25
wait_files logs/dart_more_data.done
run_round dart1000 1000 1 "data/dart_stage{stage}.jpk.npz data/dart_stage{stage}_more.jpk.npz"
variant dart1000 k25 --n-action-steps 25
wait_files logs/dart_data.done
run_round dart200 200 3 "data/dart_stage{stage}.jpk.npz"
wait
say "queue v5 finished"
