#!/usr/bin/env bash
# Experiment queue v2: demonstrations 100 / 200 / 500 / 1000 per stage (nested subsets), then DART-200.
# Low priority: the GPU is shared with the PiPER experiment.
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

train3() {  # name, episodes
  local name=$1 eps=$2 data=${3:-"data/stage{stage}.hdf5 data/stage{stage}_more.hdf5"}
  local pids=()
  for s in 1 2 3; do
    if [ -f "runs/s${s}_${name}/ckpt_$(printf %06d $STEPS)/model.safetensors" ]; then continue; fi
    $PY train_act.py --stage $s --episodes $eps --steps $STEPS --save-every $SAVE \
        --data $data --out runs/s${s}_${name} > logs/train_s${s}_${name}.log 2>&1 &
    pids+=($!)
  done
  [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
}

evalset() {  # name, step, extra args...
  local name=$1 step=$2; shift 2
  local c=$(printf %06d $step)
  $PY eval_act.py --ckpt runs/s1_${name}/ckpt_$c runs/s2_${name}/ckpt_$c runs/s3_${name}/ckpt_$c \
      --trials $TRIALS "$@" > logs/eval_${name}_${c}${EVAL_TAG:-}.log 2>&1
}

say "queue v2 start (steps=$STEPS save=$SAVE trials=$TRIALS)"
wait_files data/stage1_gen.json data/stage2_gen.json data/stage3_gen.json
say "200-demo data ready"

# CPU side: 800 more demos per stage (nested 500/1000 sets), then DART demos
( pids=()
  for s in 1 2 3; do
    [ -f data/stage${s}_more_gen.json ] || { $PYLOW gen_data.py --stage $s --episodes 800 --seed 5000 \
        --name stage${s}_more --out data > logs/gen_more_stage${s}.log 2>&1 & pids+=($!); }
  done
  [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
  echo done > logs/more_data.done
  pids=()
  for s in 1 2 3; do
    [ -f data/dart_stage${s}_gen.json ] || { $PYLOW gen_data.py --stage $s --episodes 200 --seed 3000 \
        --dart-sigma 0.01 --name dart_stage${s} --out data > logs/gen_dart_stage${s}.log 2>&1 & pids+=($!); }
  done
  [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
  echo done > logs/dart_data.done ) &

run_round() {  # name, episodes, [data]
  say "train $1"
  train3 "$@"
  say "$1 trained"
  ( evalset $1 $STEPS --out results/eval/$1 --gifs 3
    for mid in $((STEPS / 3)) $((2 * STEPS / 3)); do
      EVAL_TAG=_mid evalset $1 $mid --out results/eval/$1_ckpt$mid --gifs 0 --mode per_stage --trials 30
    done
    say "eval $1 done" ) &
}

run_round n100 100 "data/stage{stage}.hdf5"
run_round n200 200 "data/stage{stage}.hdf5"
( EVAL_TAG=_te evalset n200 $STEPS --out results/eval/n200_te --gifs 2 --temporal-ensemble 0.01; say "eval n200_te done" ) &
wait_files logs/more_data.done
run_round n500 500
run_round n1000 1000
wait_files logs/dart_data.done
run_round dart200 200 "data/dart_stage{stage}.hdf5"
wait
say "queue v2 finished"
