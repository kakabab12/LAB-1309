#!/usr/bin/env bash
# v7 demonstrations = final design + WIDER domain randomisation (dr-level 2): light 0.25-2.5x with colour tint,
# table plain colour or random pattern texture (checker/stripes/blotches); stage 2 uses the v8 side-gap rule;
# plus the v6 items below:
# light strength/direction, table colour, camera mounts (top +-12 mm/2.5 deg, wrist +-3 mm/2.5 deg),
# up to 3 non-blue distractors, pick range +-28 mm / +-30 deg. The bin colour (blue) is never changed.
# 1000 per stage in parts of 200, one worker per stage.
# Launch:  systemd-run --user --unit=act-gen7 -p MemoryMax=4G -p Nice=15 -p WorkingDirectory=$PWD \
#            --setenv=MUJOCO_GL=egl --setenv=MALLOC_MMAP_THRESHOLD_=1048576 /bin/bash $PWD/gen_v7.sh
set -u
cd "$(dirname "$0")"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> logs/queue.log; }
worker() {  # stage, design
  local s=$1 design=$2 files=()
  for i in 0 1 2 3 4; do
    local name=v7_stage${s}_p$i
    [ -f data/${name}_gen.json ] || ACT_DESIGN=$design nice -n 15 .venv/bin/python gen_data.py --stage $s --episodes 200 \
        --seed $((31000 + i)) --name $name --out data --dr --dr-level 2 > logs/gen_${name}.log 2>&1
    files+=(data/${name}.hdf5)
  done
  [ -f data/v7_stage$s.jpk.npz ] || { nice -n 15 .venv/bin/python convert_jpeg.py --out data/v7_stage$s.jpk.npz "${files[@]}" \
      >> logs/gen_v7_stage$s.log 2>&1 && rm -f "${files[@]}" && say "gen v7: v7_stage$s ready"; }
}
say "gen v7 start (domain randomisation, 1000 per stage)"
worker 1 v4 & sleep 20; worker 2 v8 & sleep 20; worker 3 v5 &
wait
for s in 1 2 3; do [ -f data/v7_stage$s.jpk.npz ] || say "gen v7 MISSING data/v7_stage$s.jpk.npz"; done
say "gen v7 done"
