#!/usr/bin/env bash
# v6 demonstrations = final grasp/place design (stages 1-2: v4, stage 3: v5) + domain randomisation:
# light strength/direction, table colour, camera mounts (top +-12 mm/2.5 deg, wrist +-3 mm/2.5 deg),
# up to 3 non-blue distractors, pick range +-28 mm / +-30 deg. The bin colour (blue) is never changed.
# 1000 per stage in parts of 200, one worker per stage.
# Launch:  systemd-run --user --unit=act-gen6 -p MemoryMax=4G -p Nice=15 -p WorkingDirectory=$PWD \
#            --setenv=MUJOCO_GL=egl --setenv=MALLOC_MMAP_THRESHOLD_=1048576 /bin/bash $PWD/gen_v6.sh
set -u
cd "$(dirname "$0")"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> logs/queue.log; }
worker() {  # stage, design
  local s=$1 design=$2 files=()
  for i in 0 1 2 3 4; do
    local name=v6_stage${s}_p$i
    [ -f data/${name}_gen.json ] || ACT_DESIGN=$design nice -n 15 .venv/bin/python gen_data.py --stage $s --episodes 200 \
        --seed $((11000 + i)) --name $name --out data --dr > logs/gen_${name}.log 2>&1
    files+=(data/${name}.hdf5)
  done
  [ -f data/v6_stage$s.jpk.npz ] || { nice -n 15 .venv/bin/python convert_jpeg.py --out data/v6_stage$s.jpk.npz "${files[@]}" \
      >> logs/gen_v6_stage$s.log 2>&1 && rm -f "${files[@]}" && say "gen v6: v6_stage$s ready"; }
}
say "gen v6 start (domain randomisation, 1000 per stage)"
worker 1 v4 & sleep 20; worker 2 v4 & sleep 20; worker 3 v5 &
wait
for s in 1 2 3; do [ -f data/v6_stage$s.jpk.npz ] || say "gen v6 MISSING data/v6_stage$s.jpk.npz"; done
say "gen v6 done"
