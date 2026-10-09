#!/usr/bin/env bash
# Clutter test: stacking on a table with other physical objects around the cell (clutter_eval.py).
#  1) demonstrations v9 = v6 (domain randomisation, final grasp/place design) + physical clutter: 0-8 objects
#     12-35 mm from the arm's path, a look-alike bin and a passing object each in 30 % of the episodes;
#     1000 per stage in parts of 200, episodes where the expert moved any clutter are discarded
#  2) while they are generated / trained: zero-shot clutter evaluation of v5_n100 (final) and v6_n1000 (randomised)
#  3) train v9 (3 stage policies, 30k steps) and evaluate it on the same clutter scenes
# Launch:  systemd-run --user --unit=act-clutter -p MemoryMax=16G -p Nice=10 -p WorkingDirectory=$PWD \
#            --setenv=MUJOCO_GL=egl --setenv=MALLOC_MMAP_THRESHOLD_=1048576 /bin/bash $PWD/run_clutter.sh
set -u
cd "$(dirname "$0")"
PY="nice -n 10 .venv/bin/python"
LOG=logs/queue.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $LOG; }
n_train() { pgrep -fc "^.venv/bin/python train_act.py" || true; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
mem_avail_gb() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
TAG=v9_n1000
CONDS_A="none sparse dense"; CONDS_B="decoy moving dense_decoy"; CONDS_C="all all_vis tight"

clutter_eval() {  # tag s1 s2 s3 conds...
  local tag=$1 c1=$2 c2=$3 c3=$4; shift 4
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 $PY clutter_eval.py --ckpt $c1 $c2 $c3 --tag $tag --conds "$@" --trials 50 \
      --gifs 2 > logs/clutter_${tag}_$1.log 2>&1
}
eval_model() {  # tag s1 s2 s3: three processes, three conditions each
  clutter_eval "$@" $CONDS_A & clutter_eval "$@" $CONDS_B & clutter_eval "$@" $CONDS_C & wait
  say "clutter eval $1 done: $(cat logs/clutter_$1_*.log | grep chained | tr '\n' ' ')"
}

say "clutter test start: v9 demos (v6 + physical clutter, 1000/stage), zero-shot eval of v5_n100 and v6_n1000"
# ---- 1) demonstrations, 6 generators at a time
gen_one() {
  local s=$1 i=$2 name=v9_stage$1_p$2 design=$([ $1 = 3 ] && echo v5 || echo v4)
  [ -f data/${name}_gen.json ] || ACT_DESIGN=$design nice -n 15 .venv/bin/python gen_data.py --stage $s --episodes 200 \
      --seed $((41000 + i)) --name $name --out data --dr --clutter > logs/gen_${name}.log 2>&1
}
export -f gen_one
( for s in 1 2 3; do for i in 0 1 2 3 4; do echo "$s $i"; done; done | xargs -P 6 -L 1 bash -c 'gen_one $0 $1'
  for s in 1 2 3; do
    files=(); for i in 0 1 2 3 4; do files+=(data/v9_stage${s}_p$i.hdf5); done
    [ -f data/v9_stage$s.jpk.npz ] || { nice -n 15 .venv/bin/python convert_jpeg.py --out data/v9_stage$s.jpk.npz "${files[@]}" \
        >> logs/gen_v9_stage$s.log 2>&1 && rm -f "${files[@]}"; }
  done
  say "gen v9: $(for s in 1 2 3; do python3 -c "
import json; k = t = 0
for i in range(5):
    d = json.load(open('data/v9_stage${s}_p%d_gen.json' % i))['summary']; k += d['kept']; t += d['tried']
print('stage $s %d/%d' % (k, t), end='; ')"; done)" ) &
GEN=$!

# ---- 2) zero-shot evaluation (starts once the generators are done, the CPU has 8 cores)
wait $GEN
for s in 1 2 3; do [ -f data/v9_stage$s.jpk.npz ] || { say "clutter: data/v9_stage$s.jpk.npz MISSING"; exit 1; }; done
( eval_model v5_n100 runs/s1_v4_n100/ckpt_030000 runs/s2_v4_n100/ckpt_030000 runs/s3_v5_n100/ckpt_030000
  eval_model v6_n1000 runs/s1_v6_n1000/ckpt_030000 runs/s2_v6_n1000/ckpt_030000 runs/s3_v6_n1000/ckpt_030000 ) &
ZS=$!

# ---- 3) train v9 on the GPU meanwhile
pids=()
for s in 1 2 3; do
  run=s${s}_$TAG
  [ -f runs/$run/ckpt_030000/model.safetensors ] && continue
  while [ "$(n_train)" -ge 5 ] || [ "$(gpu_free)" -lt 2600 ] || [ "$(mem_avail_gb)" -lt 6 ]; do sleep 30; done
  say "train $run (v9: v6 + physical clutter, 1000 demos)"
  $PY train_act.py --stage $s --episodes 1000 --steps 30000 --save-every 10000 --data data/v9_stage$s.jpk.npz \
      --out runs/$run >> logs/train_$run.log 2>&1 &
  pids+=($!)
  sleep 60
done
[ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
for s in 1 2 3; do [ -f runs/s${s}_$TAG/ckpt_030000/model.safetensors ] || { say "clutter: s${s}_$TAG not trained"; exit 1; }; done
say "v9 trained"
wait $ZS
eval_model $TAG runs/s1_$TAG/ckpt_030000 runs/s2_$TAG/ckpt_030000 runs/s3_$TAG/ckpt_030000
say "clutter test finished"
