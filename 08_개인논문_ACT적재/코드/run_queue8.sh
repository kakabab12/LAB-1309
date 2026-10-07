#!/usr/bin/env bash
# Experiment queue v8: final demo design = v4 for stages 1-2 (one fixed grasp wall) + v5 for stage 3
# (also backs off the wall before lifting, so bin C is not tipped off bin A's rims).
# Stage policies are trained independently, so stages 1-2 reuse the v4 runs and only stage 3 is new:
#   eval set v5_nN = runs/s1_v4_nN + runs/s2_v4_nN + runs/s3_v5_nN
# Training jobs start in order whenever a GPU slot is free (<= MAXT trainings on the PC incl. other queues,
# >= 2.6 GB free GPU memory, <= 2 large (1000-demo) jobs for RAM). Evaluations run one at a time, in order.
# Launch:  systemd-run --user --unit=act-queue8 -p MemoryMax=10G -p Nice=10 -p IOWeight=50 -p WorkingDirectory=$PWD \
#            --setenv=MUJOCO_GL=egl --setenv=MALLOC_MMAP_THRESHOLD_=1048576 /bin/bash $PWD/run_queue8.sh
set -u
cd "$(dirname "$0")"
PY="nice -n 10 .venv/bin/python"
STEPS=${STEPS:-30000}
SAVE=${SAVE:-10000}
TRIALS=${TRIALS:-50}
MAXT=${MAXT:-4}
LOG=logs/queue.log
mkdir -p logs runs results/eval
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
wait_files() { for f in "$@"; do until [ -f "$f" ]; do sleep 30; done; done; }
ckpt() { echo "runs/$1/ckpt_$(printf %06d ${2:-$STEPS})"; }
done_run() { [ -f "$(ckpt $1)/model.safetensors" ]; }
n_train() { pgrep -fc "python train_act.py" || true; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
BIG_PIDS=()
n_big() { local n=0; for p in "${BIG_PIDS[@]:-}"; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && n=$((n + 1)); done; echo $n; }

train_job() {  # run, stage, episodes, data files...   (runs in background, one retry after a crash)
  local run=$1 s=$2 eps=$3; shift 3
  for attempt in 1 2; do
    done_run $run && return 0
    $PY train_act.py --stage $s --episodes $eps --steps $STEPS --save-every $SAVE --data "$@" --out runs/$run \
        >> logs/train_${run}.log 2>&1
    done_run $run && { say "trained $run"; return 0; }
    say "$run: training ended without a final checkpoint (attempt $attempt)"
    sleep 300
  done
}

launch() {  # run, stage, episodes, data files...
  local run=$1 eps=$3
  done_run $run && return
  wait_files "${@:4}"
  while [ "$(n_train)" -ge $MAXT ] || [ "$(gpu_free)" -lt 2600 ] || { [ $eps -ge 1000 ] && [ "$(n_big)" -ge 2 ]; }; do
    sleep 30
  done
  say "train $run ($eps demos)"
  train_job "$@" &
  [ $eps -ge 1000 ] && BIG_PIDS+=($!)
  sleep 90  # let its memory settle before the next launch decision
}

eval_worker() {  # name:run1,run2,run3 ...   one evaluation at a time, in order
  for spec in "$@"; do
    local name=${spec%%:*} runs=${spec#*:}
    local r1 r2 r3; IFS=, read -r r1 r2 r3 <<< "$runs"
    [ -f results/eval/$name/results.json ] && continue
    until done_run $r1 && done_run $r2 && done_run $r3; do sleep 60; done
    say "eval $name ($r1 + $r2 + $r3)"
    $PY eval_act.py --ckpt $(ckpt $r1) $(ckpt $r2) $(ckpt $r3) --trials $TRIALS --out results/eval/$name --gifs 3 \
        > logs/eval_${name}.log 2>&1
    say "eval $name done"
  done
}

say "queue v8 start (stages 1-2 v4 data, stage 3 v5 data; steps=$STEPS trials=$TRIALS)"
# the frozen v7 queue still owns the v4_n200 trainings and the v4_n100 mid-evals; retire it once they are done
( wait_files $(ckpt s1_v4_n200)/model.safetensors $(ckpt s2_v4_n200)/model.safetensors $(ckpt s3_v4_n200)/model.safetensors
  until grep -q "eval v4_n100 done" $LOG; do sleep 60; done
  systemctl --user stop act-queue7 && say "queue v7 retired (v4_n200 trained, v4_n100 evals done)" ) &

eval_worker "v5_n100:s1_v4_n100,s2_v4_n100,s3_v5_n100" "v5_n200:s1_v4_n200,s2_v4_n200,s3_v5_n200" \
            "v5_n500:s1_v4_n500,s2_v4_n500,s3_v5_n500" "v5_n1000:s1_v4_n1000,s2_v4_n1000,s3_v5_n1000" \
            "v5_dart1000:s1_v4_dart1000,s2_v4_dart1000,s3_v5_dart1000" \
            "v5_dart200:s1_v4_dart200,s2_v4_dart200,s3_v5_dart200" \
            "v4_n200:s1_v4_n200,s2_v4_n200,s3_v4_n200" &
EVAL=$!

V4="data/v4_stage{stage}.jpk.npz"; V4M="data/v4_stage{stage}_more.jpk.npz"
V4D="data/v4_dart_stage{stage}.jpk.npz"; V4DM="data/v4_dart_stage{stage}_more.jpk.npz"
f() { echo "${1//\{stage\}/$2}"; }  # resolve {stage} for wait_files
launch s3_v5_n100 3 100 data/v5_stage3.jpk.npz
launch s3_v5_n200 3 200 data/v5_stage3.jpk.npz
launch s1_v4_n500 1 500 $(f $V4 1) $(f $V4M 1)
launch s2_v4_n500 2 500 $(f $V4 2) $(f $V4M 2)
launch s3_v5_n500 3 500 data/v5_stage3.jpk.npz data/v5_stage3_more.jpk.npz
launch s1_v4_n1000 1 1000 $(f $V4 1) $(f $V4M 1)
launch s2_v4_n1000 2 1000 $(f $V4 2) $(f $V4M 2)
launch s3_v5_n1000 3 1000 data/v5_stage3.jpk.npz data/v5_stage3_more.jpk.npz
launch s1_v4_dart1000 1 1000 $(f $V4D 1) $(f $V4DM 1)
launch s2_v4_dart1000 2 1000 $(f $V4D 2) $(f $V4DM 2)
launch s3_v5_dart1000 3 1000 data/v5_dart_stage3.jpk.npz data/v5_dart_stage3_more.jpk.npz
launch s1_v4_dart200 1 200 $(f $V4D 1)
launch s2_v4_dart200 2 200 $(f $V4D 2)
launch s3_v5_dart200 3 200 data/v5_dart_stage3.jpk.npz
say "queue v8: all trainings launched"
wait $EVAL
say "queue v8 finished"
