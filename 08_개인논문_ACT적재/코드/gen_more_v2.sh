#!/usr/bin/env bash
# v2 "more" demonstrations in parts of 200 (a crash loses at most one part), then one pack per stage.
#   normal: stage{s}_more (seeds 5000..5003)   DART: dart_stage{s}_more (seeds 7000..7003, sigma 0.005)
cd "$(dirname "$0")"
PYLOW="nice -n 15 .venv/bin/python"
parts() {  # stage, stem, seed0, extra args...
  local s=$1 stem=$2 seed0=$3; shift 3
  local files=()
  for i in 0 1 2 3; do
    local name=${stem}_p$i
    [ -f data/${name}_gen.json ] || $PYLOW gen_data.py --stage $s --episodes 200 --seed $((seed0 + i)) --name $name --out data "$@" > logs/gen_${name}.log 2>&1
    files+=(data/${name}.hdf5)
  done
  $PYLOW convert_jpeg.py --out data/${stem}.jpk.npz "${files[@]}" >> logs/gen_${stem}.log 2>&1 && rm -f "${files[@]}"
}
# stage 1 normal "more" already exists as one complete 800-episode file
[ -f data/stage1_more.jpk.npz ] || $PYLOW convert_jpeg.py data/stage1_more.hdf5 >> logs/gen_stage1_more.log 2>&1 && rm -f data/stage1_more.hdf5
( [ -f data/stage2_more.jpk.npz ] || parts 2 stage2_more 5000 ) &
( [ -f data/stage3_more.jpk.npz ] || parts 3 stage3_more 5000 ) &
wait
echo "[$(date '+%m-%d %H:%M:%S')] v2 more demos ready" >> logs/queue.log
for s in 1 2 3; do ( [ -f data/dart_stage${s}_more.jpk.npz ] || parts $s dart_stage${s}_more 7000 --dart-sigma 0.005 ) & done
wait
echo "[$(date '+%m-%d %H:%M:%S')] v2 DART more demos ready" >> logs/queue.log
