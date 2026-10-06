#!/usr/bin/env bash
# v2 demonstrations for one stage, in the order the queue needs them: base 200 -> DART 200 -> more 800 -> DART more 800.
S=$1
cd "$(dirname "$0")"
PYLOW="nice -n 15 .venv/bin/python"
pack() {  # stem, gen args...
  local stem=$1; shift
  [ -f data/${stem}.jpk.npz ] && return
  [ -f data/${stem}_gen.json ] || $PYLOW gen_data.py --stage $S --name $stem --out data "$@" > logs/gen_${stem}.log 2>&1
  $PYLOW convert_jpeg.py data/${stem}.hdf5 >> logs/gen_${stem}.log 2>&1 && rm -f data/${stem}.hdf5
}
pack stage$S --episodes 200 --seed 1000
pack dart_stage$S --episodes 200 --seed 3000 --dart-sigma 0.005
pack stage${S}_more --episodes 800 --seed 5000
pack dart_stage${S}_more --episodes 800 --seed 7000 --dart-sigma 0.005
