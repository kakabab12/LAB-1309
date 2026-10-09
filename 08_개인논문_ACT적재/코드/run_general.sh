#!/usr/bin/env bash
# Generalisation test of the method (audit the demonstrations, fix the grasp choice with a rule):
# 3 objects (square bin, cup with handle, rectangular box) x 2 cell layouts (original, mirrored) x 2 demonstration
# designs (naive: grasp the side nearest the robot, switching between walls / sides of the handle; rule: always the same
# wall / the same side of the handle). Both designs keep the 9 mm clearance and the back-off, so only the grasp choice
# differs. 100 demonstrations per stage, ACT 30k steps, then GPU evaluation (per-stage 50 + chained 50).
set -u
cd "$(dirname "$0")"
PY="nice -n 10 .venv/bin/python"
LOG=logs/queue.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $LOG; }
n_train() { pgrep -fc "^.venv/bin/python train_act.py" || true; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
mem_avail_gb() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
COND=(  # object layout naive-design rule-design
  "bin orig v2b v5" "cup orig cupnaive cuprule" "box orig v2b v5"
  "bin mirror v2b v5" "cup mirror cupnaive cuprule" "box mirror v2b v5")
tag() { echo "g_${1}_${2}_${3}"; }  # object layout design
# the CPU is shared with the clutter test: start once its demonstrations are packed
while [ ! -f data/v9_stage3.jpk.npz ] && systemctl --user is-active --quiet act-clutter; do sleep 60; done
say "generalisation start: ${#COND[@]} object/layout conditions x 2 designs x 3 stages (100 demos each)"

# ---- 1) demonstrations (8 generator processes at a time)
jobs_gen=()
for c in "${COND[@]}"; do set -- $c; for d in $3 $4; do for st in 1 2 3; do jobs_gen+=("$1 $2 $d $st"); done; done; done
gen_one() {
  local obj=$1 lay=$2 d=$3 st=$4 name=$(tag $1 $2 $3)_s$4
  [ -f data/$name.jpk.npz ] && return 0
  ACT_OBJECT=$obj ACT_LAYOUT=$lay ACT_DESIGN=$d nice -n 15 .venv/bin/python gen_data.py --stage $st --episodes 100 \
      --seed $((70000 + 100 * st)) --name $name --out data > logs/gen_$name.log 2>&1 \
    && nice -n 15 .venv/bin/python convert_jpeg.py data/$name.hdf5 >> logs/gen_$name.log 2>&1 && rm -f data/$name.hdf5
}
export -f gen_one tag
printf '%s\n' "${jobs_gen[@]}" | xargs -P 8 -L 1 bash -c 'gen_one $0 $1 $2 $3'
miss=""; for j in "${jobs_gen[@]}"; do set -- $j; [ -f data/$(tag $1 $2 $3)_s$4.jpk.npz ] || miss="$miss $(tag $1 $2 $3)_s$4"; done
[ -n "$miss" ] && { say "generalisation: demos MISSING$miss"; exit 1; }
say "generalisation: demos ready -> audit"
$PY demo_audit_general.py > logs/demo_audit_general.log 2>&1
say "generalisation audit: $(tail -1 logs/demo_audit_general.log)"

# ---- 2) training (up to 5 at a time) and 3) evaluation as soon as a design's 3 stages are done
evaluate() {  # obj lay design
  local t=$(tag $1 $2 $3)
  [ -f results/eval/$t/results.json ] && return 0
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 ACT_OBJECT=$1 ACT_LAYOUT=$2 ACT_DESIGN=$3 $PY eval_act.py --device cpu --ckpt runs/${t}_s1/ckpt_030000 runs/${t}_s2/ckpt_030000 \
      runs/${t}_s3/ckpt_030000 --trials 50 --gifs 2 --out results/eval/$t > logs/eval_$t.log 2>&1
  say "eval $t done: $(grep '\[chained\]' logs/eval_$t.log)"
}
pids=()
for c in "${COND[@]}"; do set -- $c; obj=$1; lay=$2
  for d in $3 $4; do t=$(tag $obj $lay $d)
    for st in 1 2 3; do
      run=${t}_s$st
      [ -f runs/$run/ckpt_030000/model.safetensors ] && continue
      while [ "$(n_train)" -ge 5 ] || [ "$(gpu_free)" -lt 2600 ] || [ "$(mem_avail_gb)" -lt 6 ]; do sleep 30; done
      say "train $run"
      ACT_OBJECT=$obj ACT_LAYOUT=$lay $PY train_act.py --stage $st --episodes 100 --steps 30000 --save-every 10000 \
          --data data/$run.jpk.npz --out runs/$run >> logs/train_$run.log 2>&1 &
      pids+=($!)
      sleep 60
    done
    ( while [ ! -f runs/${t}_s1/ckpt_030000/model.safetensors ] || [ ! -f runs/${t}_s2/ckpt_030000/model.safetensors ] \
            || [ ! -f runs/${t}_s3/ckpt_030000/model.safetensors ]; do sleep 120; done
      evaluate $obj $lay $d ) &
  done
done
wait
say "generalisation finished"
