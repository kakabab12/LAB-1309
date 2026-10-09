#!/usr/bin/env bash
# Integrated test: ONE policy per stage for every object (square bin, 6-facet cup with handle, rectangular box) in
# both cell layouts (original, mirrored), inside a busy factory cell with randomised looks (factory.py).
#  1) demonstrations with the rule design of each object (bin / box: same wall + back-off = v5, cup: same side of
#     the handle = cuprule), domain randomisation (v6 or v7 ranges) + 15 % plain / 25 % table clutter / 60 % factory
#     scenes (gen_data.py --dr --factory, keep-out map per object and layout from clutter_map.py), 300 per
#     object/layout/stage -> 1800 per stage
#  2) train s{1,2,3}_int_n1800 on all of them (40k steps, checkpoints every 10k)
#  3) evaluate every object/layout on none / all / factory / factory_vis / factory_max (clutter_eval.py)
# Launch:  systemd-run --user --unit=act-integrated -p MemoryMax=16G -p Nice=10 -p WorkingDirectory=$PWD \
#            --setenv=MUJOCO_GL=egl --setenv=MALLOC_MMAP_THRESHOLD_=1048576 /bin/bash $PWD/run_integrated.sh
set -u
cd "$(dirname "$0")"
PY="nice -n 10 .venv/bin/python"
LOG=logs/queue.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $LOG; }
n_train() { pgrep -fc "^.venv/bin/python train_act.py" || true; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
mem_avail_gb() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
COMBOS=("bin orig v5" "bin mirror v5" "cup orig cuprule" "cup mirror cuprule" "box orig v5" "box mirror v5")
TAG=int_n1800
STEPS=40000  # 60k planned; 40k to finish by the weekend (10/11)

# keep-out maps for every object / layout, and the CPU free of the generalisation test's demo generation
for c in "${COMBOS[@]}"; do set -- $c; s=$([ "$1" = bin ] && [ "$2" = orig ] && echo "" || echo "_$1$([ "$2" = mirror ] && echo _mirror)")
  s=${s/_bin_mirror/_mirror}
  until [ -f results/clutter_keepout$s.npz ] && python3 -c "import numpy as np, sys; sys.exit('zmax' not in np.load('results/clutter_keepout$s.npz').files)"; do sleep 60; done; done
while ! grep -q "generalisation: demos ready" $LOG && systemctl --user is-active --quiet act-general; do sleep 120; done
say "integrated test start: 3 objects x 2 layouts, rule designs + randomisation + factory cell, 300 demos each per stage"

# ---- 1) demonstrations
gen_one() {  # obj lay design stage part
  local name=i_$1_$2_s$4_p$5
  [ -f data/$name.jpk.npz ] && return 0
  ACT_OBJECT=$1 ACT_LAYOUT=$2 ACT_DESIGN=$3 nice -n 15 .venv/bin/python gen_data.py --stage $4 --episodes 100 \
      --seed $((51000 + 10 * $5)) --name $name --out data --dr --factory > logs/gen_$name.log 2>&1 \
    && nice -n 15 .venv/bin/python convert_jpeg.py data/$name.hdf5 >> logs/gen_$name.log 2>&1 && rm -f data/$name.hdf5
}
export -f gen_one
for c in "${COMBOS[@]}"; do for st in 1 2 3; do for p in 0 1 2; do echo "$c $st $p"; done; done; done \
  | xargs -P 6 -L 1 bash -c 'gen_one $0 $1 $2 $3 $4'
miss=0; for c in "${COMBOS[@]}"; do set -- $c; for st in 1 2 3; do for p in 0 1 2; do
  [ -f data/i_$1_$2_s${st}_p$p.jpk.npz ] || { miss=1; say "integrated: data/i_$1_$2_s${st}_p$p.jpk.npz MISSING"; }; done; done; done
[ $miss = 0 ] || exit 1
say "integrated: demos ready"

# ---- 2) one policy per stage on everything
pids=()
for st in 1 2 3; do
  run=s${st}_$TAG
  [ -f runs/$run/ckpt_$(printf %06d $STEPS)/model.safetensors ] && continue
  packs=(); for c in "${COMBOS[@]}"; do set -- $c; for p in 0 1 2; do packs+=(data/i_$1_$2_s${st}_p$p.jpk.npz); done; done
  while [ "$(n_train)" -ge 5 ] || [ "$(gpu_free)" -lt 2600 ] || [ "$(mem_avail_gb)" -lt 6 ]; do sleep 30; done
  say "train $run (all objects/layouts + factory cell, 1800 demos, $STEPS steps)"
  $PY train_act.py --stage $st --episodes 1800 --steps $STEPS --save-every 10000 --data "${packs[@]}" \
      --out runs/$run >> logs/train_$run.log 2>&1 &
  pids+=($!)
  sleep 60
done
[ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
ck=ckpt_$(printf %06d $STEPS)
for st in 1 2 3; do [ -f runs/s${st}_$TAG/$ck/model.safetensors ] || { say "integrated: s${st}_$TAG not trained"; exit 1; }; done
say "integrated policies trained"

# ---- 3) every object / layout on the clutter conditions, 6 processes
ev() {  # obj lay design
  ACT_OBJECT=$1 ACT_LAYOUT=$2 ACT_DESIGN=$3 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 $PY clutter_eval.py \
      --ckpt runs/s1_$TAG/$ck runs/s2_$TAG/$ck runs/s3_$TAG/$ck --tag ${TAG}_$1_$2 --conds none all factory factory_vis factory_max --factory-env \
      --mode chained --trials 50 --gifs 2 > logs/clutter_${TAG}_$1_$2.log 2>&1
  say "integrated eval $1 $2: $(grep chained logs/clutter_${TAG}_$1_$2.log | tr '\n' ' ')"
}
for c in "${COMBOS[@]}"; do ev $c & done
wait
say "integrated test finished"
