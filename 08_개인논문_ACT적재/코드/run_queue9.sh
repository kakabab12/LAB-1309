#!/usr/bin/env bash
# Experiment queue v9: every remaining training job in one priority list, started as soon as its data exists
# and a GPU slot is free (<= MAXT trainings on the PC, >= 2.6 GB free GPU memory, >= 5 GB MemAvailable).
# Packs are memory-mapped by train_act.py, so 1000-demo jobs no longer need their own RAM copy.
#   v4/v5  : final grasp/place design (stages 1-2 v4 data, stage 3 v5 data)
#   v6     : same design + domain randomisation (lights, table colour, camera mounts, distractors, wider range)
#   ...c   : brightness normalisation + CLAHE on the camera images (train and eval)
# The v5_* evaluation sets are run by queue v8's evaluator (still alive, its launcher is frozen);
# this queue evaluates the new sets.
# Launch:  systemd-run --user --unit=act-queue9 -p MemoryMax=12G -p Nice=10 -p IOWeight=50 -p WorkingDirectory=$PWD \
#            --setenv=MUJOCO_GL=egl --setenv=MALLOC_MMAP_THRESHOLD_=1048576 /bin/bash $PWD/run_queue9.sh
set -u
cd "$(dirname "$0")"
PY="nice -n 10 .venv/bin/python"
STEPS=${STEPS:-30000}
SAVE=${SAVE:-10000}
TRIALS=${TRIALS:-50}
MAXT=${MAXT:-4}
LOG=logs/queue.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
ckpt() { echo "runs/$1/ckpt_$(printf %06d $STEPS)"; }
done_run() { [ -f "$(ckpt $1)/model.safetensors" ]; }
n_train() { pgrep -fc "^.venv/bin/python train_act.py" || true; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
mem_avail_gb() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
running() { pgrep -f -- "--out runs/$1\$" > /dev/null || pgrep -f -- "--out runs/$1 " > /dev/null; }

# run | stage | episodes | preprocess (- = none) | data files      (priority order)
V4=data/v4_stage; V4D=data/v4_dart_stage
JOBS=(
  "s3_v5_n200|3|200|-|data/v5_stage3.jpk.npz"
  "s1_v6c_n1000|1|1000|clahe|data/v6_stage1.jpk.npz"
  "s2_v6c_n1000|2|1000|clahe|data/v6_stage2.jpk.npz"
  "s3_v6c_n1000|3|1000|clahe|data/v6_stage3.jpk.npz"
  "s1_v4_n500|1|500|-|${V4}1.jpk.npz ${V4}1_more.jpk.npz"
  "s2_v4_n500|2|500|-|${V4}2.jpk.npz ${V4}2_more.jpk.npz"
  "s3_v5_n500|3|500|-|data/v5_stage3.jpk.npz data/v5_stage3_more.jpk.npz"
  "s1_v4_n1000|1|1000|-|${V4}1.jpk.npz ${V4}1_more.jpk.npz"
  "s2_v4_n1000|2|1000|-|${V4}2.jpk.npz ${V4}2_more.jpk.npz"
  "s3_v5_n1000|3|1000|-|data/v5_stage3.jpk.npz data/v5_stage3_more.jpk.npz"
  "s1_v6_n1000|1|1000|-|data/v6_stage1.jpk.npz"
  "s2_v6_n1000|2|1000|-|data/v6_stage2.jpk.npz"
  "s3_v6_n1000|3|1000|-|data/v6_stage3.jpk.npz"
  "s1_v4c_n200|1|200|clahe|${V4}1.jpk.npz"
  "s2_v4c_n200|2|200|clahe|${V4}2.jpk.npz"
  "s3_v5c_n200|3|200|clahe|data/v5_stage3.jpk.npz"
  "s1_v4_dart1000|1|1000|-|${V4D}1.jpk.npz ${V4D}1_more.jpk.npz"
  "s2_v4_dart1000|2|1000|-|${V4D}2.jpk.npz ${V4D}2_more.jpk.npz"
  "s3_v5_dart1000|3|1000|-|data/v5_dart_stage3.jpk.npz data/v5_dart_stage3_more.jpk.npz"
  "s1_v4_dart200|1|200|-|${V4D}1.jpk.npz"
  "s2_v4_dart200|2|200|-|${V4D}2.jpk.npz"
  "s3_v5_dart200|3|200|-|data/v5_dart_stage3.jpk.npz"
)

train_job() {  # run stage eps pre files   (one retry after a crash)
  local run=$1 s=$2 eps=$3 pre=$4 files=$5 extra=()
  [ "$pre" != "-" ] && extra=(--preprocess $pre)
  for attempt in 1 2; do
    done_run $run && return 0
    $PY train_act.py --stage $s --episodes $eps --steps $STEPS --save-every $SAVE --data $files "${extra[@]}" \
        --out runs/$run >> logs/train_${run}.log 2>&1
    done_run $run && { say "trained $run"; return 0; }
    say "$run: training ended without a final checkpoint (attempt $attempt)"
    sleep 300
  done
}

eval_worker() {  # name:run1,run2,run3 ...
  for spec in "$@"; do
    local name=${spec%%:*} r1 r2 r3
    IFS=, read -r r1 r2 r3 <<< "${spec#*:}"
    [ -f results/eval/$name/results.json ] && continue
    until done_run $r1 && done_run $r2 && done_run $r3; do sleep 60; done
    say "eval $name ($r1 + $r2 + $r3)"
    $PY eval_act.py --ckpt $(ckpt $r1) $(ckpt $r2) $(ckpt $r3) --trials $TRIALS --out results/eval/$name --gifs 3 \
        > logs/eval_${name}.log 2>&1
    say "eval $name done"
  done
}

say "queue v9 start (${#JOBS[@]} training jobs in priority order, MAXT=$MAXT)"
eval_worker "v6c_n1000:s1_v6c_n1000,s2_v6c_n1000,s3_v6c_n1000" "v6_n1000:s1_v6_n1000,s2_v6_n1000,s3_v6_n1000" \
            "v5c_n200:s1_v4c_n200,s2_v4c_n200,s3_v5c_n200" "v4_n200:s1_v4_n200,s2_v4_n200,s3_v4_n200" &
EVAL=$!

declare -A LAUNCHED=()
while :; do
  pending=0 started=""
  for job in "${JOBS[@]}"; do
    IFS='|' read -r run s eps pre files <<< "$job"
    { [ -n "${LAUNCHED[$run]:-}" ] || done_run $run; } && continue
    pending=$((pending + 1))
    running $run && { LAUNCHED[$run]=1; continue; }  # e.g. started by queue v8 before it was frozen
    ok=1; for f in $files; do [ -f "$f" ] || ok=0; done
    [ $ok = 1 ] || continue
    if [ "$(n_train)" -lt $MAXT ] && [ "$(gpu_free)" -ge 2600 ] && [ "$(mem_avail_gb)" -ge 5 ]; then
      say "train $run ($eps demos, preprocess $pre)"
      train_job "$run" "$s" "$eps" "$pre" "$files" &
      LAUNCHED[$run]=1 started=$run
      break
    fi
    break  # highest-priority ready job waits for a slot; do not let lower ones jump ahead
  done
  [ $pending = 0 ] && break
  [ -n "$started" ] && sleep 90 || sleep 30
done
say "queue v9: all trainings launched"
wait
say "queue v9 finished"
