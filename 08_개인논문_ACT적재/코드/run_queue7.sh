#!/usr/bin/env bash
# Experiment queue v7: v4 demonstrations (one fixed grasp wall; data from gen_v4.sh in unit act-gen4).
# Same order as v6: n100 -> n200 -> n500 -> n1000 -> DART1000 -> DART200, same eval seeds as v1/v2.
# Evaluation of a round runs in the background while the next round trains (GPU has room for 3 trainings + 1 eval).
# Launch:  systemd-run --user --unit=act-queue7 -p MemoryMax=8G -p Nice=10 -p IOWeight=50 -p WorkingDirectory=$PWD \
#            --setenv=MUJOCO_GL=egl --setenv=ACT_DESIGN=v4 --setenv=MALLOC_MMAP_THRESHOLD_=1048576 /bin/bash $PWD/run_queue7.sh
set -u
cd "$(dirname "$0")"
PY="nice -n 10 .venv/bin/python"
STEPS=${STEPS:-30000}
SAVE=${SAVE:-10000}
TRIALS=${TRIALS:-50}
LOG=logs/queue.log
mkdir -p logs runs results/eval
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
wait_files() { for f in "$@"; do until [ -f "$f" ]; do sleep 30; done; done; }
final_ckpt() { echo "runs/s${1}_${2}/ckpt_$(printf %06d $STEPS)/model.safetensors"; }

train_round() {  # name, episodes, max_parallel, data...
  local name=$1 eps=$2 maxpar=$3; shift 3
  local data="$*"
  for attempt in 1 2; do  # a CUDA OOM (GPU shared with PiPER) loses one stage; retry it once
    local pids=()
    for s in 1 2 3; do
      [ -f "$(final_ckpt $s $name)" ] && continue
      while [ "$(jobs -rp | grep -c -F -f <(printf '%s\n' "${pids[@]:-none}"))" -ge "$maxpar" ]; do sleep 20; done
      $PY train_act.py --stage $s --episodes $eps --steps $STEPS --save-every $SAVE \
          --data $data --out runs/s${s}_${name} >> logs/train_s${s}_${name}.log 2>&1 &
      pids+=($!)
      sleep 60  # stagger start-up memory peaks
    done
    [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
    local missing=0
    for s in 1 2 3; do [ -f "$(final_ckpt $s $name)" ] || missing=1; done
    [ $missing = 0 ] && return 0
    say "$name: training incomplete (attempt $attempt), retrying missing stages"
    sleep 300
  done
  return 1
}

evalset() {  # name, step, extra args...
  local name=$1 step=$2; shift 2
  local c=$(printf %06d $step)
  $PY eval_act.py --ckpt runs/s1_${name}/ckpt_$c runs/s2_${name}/ckpt_$c runs/s3_${name}/ckpt_$c \
      --trials $TRIALS "$@" > logs/eval_${name}_${c}${EVAL_TAG:-}.log 2>&1
}

evals() {  # name: final eval (+ learning-curve evals for the small rounds)
  local name=$1
  [ -f results/eval/$name/results.json ] || evalset $name $STEPS --out results/eval/$name --gifs 3
  case $name in v4_n100|v4_n200)
    for mid in $((STEPS / 3)) $((2 * STEPS / 3)); do
      [ -f results/eval/${name}_ckpt$mid/results.json ] || \
        EVAL_TAG=_mid evalset $name $mid --out results/eval/${name}_ckpt$mid --gifs 0 --mode per_stage --trials 30
    done;;
  esac
  say "eval $name done"
}

EVAL_PIDS=()
run_round() {  # name, episodes, max_parallel, data...
  local name=$1
  say "train $name"
  if train_round "$@"; then
    say "$name trained; evaluating in background"
    evals $name &
    EVAL_PIDS+=($!)
  else
    say "$name FAILED to train; skipping its evaluation"
  fi
}

say "queue v7 start (v4 demos, steps=$STEPS trials=$TRIALS)"
packs() { for s in 1 2 3; do echo data/${1}${s}${2}.jpk.npz; done; }
# v6 still owns the GPU until its n200 training ends; afterwards it only evaluates, then idles waiting for v2 data
until sed -n '/queue v6 start/,$p' $LOG | grep -q "n200 trained"; do sleep 60; done
( until grep -q "eval n200_te done" $LOG; do sleep 60; done; sleep 10
  systemctl --user stop act-queue && say "queue v6 stopped after n200_te (v2 n500+ replaced by v4)" ) &
wait_files $(packs v4_stage "")
run_round v4_n100 100 3 "data/v4_stage{stage}.jpk.npz"
run_round v4_n200 200 3 "data/v4_stage{stage}.jpk.npz"
wait_files $(packs v4_stage _more)
run_round v4_n500 500 2 "data/v4_stage{stage}.jpk.npz data/v4_stage{stage}_more.jpk.npz"
run_round v4_n1000 1000 1 "data/v4_stage{stage}.jpk.npz data/v4_stage{stage}_more.jpk.npz"
wait_files $(packs v4_dart_stage _more)
run_round v4_dart1000 1000 1 "data/v4_dart_stage{stage}.jpk.npz data/v4_dart_stage{stage}_more.jpk.npz"
run_round v4_dart200 200 3 "data/v4_dart_stage{stage}.jpk.npz"
wait "${EVAL_PIDS[@]}"
say "queue v7 finished"
