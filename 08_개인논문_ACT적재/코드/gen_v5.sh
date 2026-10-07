#!/usr/bin/env bash
# v5 stage-3 demonstrations (v4 + back off 5 mm from the wall before lifting after release).
# Stages 1-2 keep the v4 data: on the floor the straight lift never tipped a bin (v4 n100: 100 / 100%).
# Parts of 200 in parallel, packs converted as soon as their parts exist (base first: training waits on it).
# Launch:  systemd-run --user --unit=act-gen5 -p MemoryMax=4G -p Nice=15 -p WorkingDirectory=$PWD \
#            --setenv=MUJOCO_GL=egl --setenv=ACT_DESIGN=v5 --setenv=MALLOC_MMAP_THRESHOLD_=1048576 /bin/bash $PWD/gen_v5.sh
set -u
cd "$(dirname "$0")"
[ "${ACT_DESIGN:-}" = v5 ] || { echo "ACT_DESIGN=v5 required"; exit 1; }
PYLOW="nice -n 15 .venv/bin/python"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> logs/queue.log; }
part() {  # name, seed, extra args
  local name=$1 seed=$2; shift 2
  [ -f data/${name}_gen.json ] || $PYLOW gen_data.py --stage 3 --episodes 200 --seed $seed --name $name --out data "$@" > logs/gen_${name}.log 2>&1
}
pack() {  # stem, part names...
  local stem=$1; shift
  [ -f data/${stem}.jpk.npz ] && return
  local files=(); for p in "$@"; do files+=(data/${p}.hdf5); done
  $PYLOW convert_jpeg.py --out data/${stem}.jpk.npz "${files[@]}" >> logs/gen_${stem}.log 2>&1 && rm -f "${files[@]}" && say "gen v5: ${stem} ready"
}
D="--dart-sigma 0.005"
say "gen v5 start (stage 3)"
( part v5_stage3_p0 1000; pack v5_stage3 v5_stage3_p0 ) &
part v5_stage3_more_p0 5000 & part v5_stage3_more_p1 5001 &
wait
part v5_stage3_more_p2 5002 & part v5_stage3_more_p3 5003 & part v5_dart_stage3_p0 3000 $D &
wait
pack v5_stage3_more v5_stage3_more_p0 v5_stage3_more_p1 v5_stage3_more_p2 v5_stage3_more_p3 &
pack v5_dart_stage3 v5_dart_stage3_p0 &
for i in 0 1 2 3; do part v5_dart_stage3_more_p$i $((7000 + i)) $D & done
wait
pack v5_dart_stage3_more v5_dart_stage3_more_p0 v5_dart_stage3_more_p1 v5_dart_stage3_more_p2 v5_dart_stage3_more_p3
for f in v5_stage3 v5_stage3_more v5_dart_stage3 v5_dart_stage3_more; do [ -f data/$f.jpk.npz ] || say "gen v5 MISSING data/$f.jpk.npz"; done
say "gen v5 done"
