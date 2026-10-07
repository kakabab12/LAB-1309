#!/usr/bin/env bash
# v4 demonstrations (one fixed grasp wall over the whole yaw range; 9 mm clearance as v2), one worker per stage,
# in the order the queue needs them: base 200 -> more 4x200 -> DART 200 -> DART more 4x200.
# Parts of 200 so a crash loses at most one part; the streaming converter joins parts into one pack.
# Launch:  systemd-run --user --unit=act-gen4 -p MemoryMax=5G -p Nice=15 -p WorkingDirectory=$PWD \
#            --setenv=MUJOCO_GL=egl --setenv=ACT_DESIGN=v4 --setenv=MALLOC_MMAP_THRESHOLD_=1048576 /bin/bash $PWD/gen_v4.sh
set -u
cd "$(dirname "$0")"
[ "${ACT_DESIGN:-}" = v4 ] || { echo "ACT_DESIGN=v4 required"; exit 1; }
PYLOW="nice -n 15 .venv/bin/python"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> logs/queue.log; }

pack() {  # stage, stem, seed0, nparts, extra gen args...
  local s=$1 stem=$2 seed0=$3 np=$4; shift 4
  [ -f data/${stem}.jpk.npz ] && return
  local files=()
  for ((i = 0; i < np; i++)); do
    local name=${stem}_p$i
    [ -f data/${name}_gen.json ] || $PYLOW gen_data.py --stage $s --episodes 200 --seed $((seed0 + i)) --name $name --out data "$@" > logs/gen_${name}.log 2>&1
    files+=(data/${name}.hdf5)
  done
  $PYLOW convert_jpeg.py --out data/${stem}.jpk.npz "${files[@]}" >> logs/gen_${stem}.log 2>&1 && rm -f "${files[@]}"
}

worker() {  # stage
  local s=$1
  pack $s v4_stage$s 1000 1
  pack $s v4_stage${s}_more 5000 4
  pack $s v4_dart_stage$s 3000 1 --dart-sigma 0.005
  pack $s v4_dart_stage${s}_more 7000 4 --dart-sigma 0.005
}

say "gen v4 start"
for s in 1 2 3; do worker $s & sleep 20; done
wait
for s in 1 2 3; do for k in stage stage_more dart_stage dart_stage_more; do
  f=data/v4_${k/stage/stage$s}.jpk.npz; [ -f $f ] || say "gen v4 MISSING $f"
done; done
say "gen v4 done"
