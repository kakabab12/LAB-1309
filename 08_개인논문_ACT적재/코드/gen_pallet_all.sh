#!/usr/bin/env bash
# 2x2 x 2-layer pallet demonstrations: 100 per slot, 4 workers (slot pairs balance the slow upper slots),
# then one JPEG pack per slot (data/pallet_s1..s8.jpk.npz).
# Launch:  systemd-run --user --unit=act-genpal -p MemoryMax=6G -p Nice=15 -p WorkingDirectory=$PWD \
#            --setenv=MUJOCO_GL=egl --setenv=ACT_DESIGN=v4 --setenv=PALLET_VIA=far /bin/bash $PWD/gen_pallet_all.sh
set -u
cd "$(dirname "$0")"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> logs/queue.log; }
one() {  # 0-based slot
  local k=$1 name=pallet_s$(($1 + 1))
  [ -f data/$name.jpk.npz ] && return
  [ -f data/${name}_gen.json ] || nice -n 15 .venv/bin/python gen_pallet.py --slot $k --episodes 100 --seed 21000 --name $name \
      --out data > logs/gen_$name.log 2>&1
  nice -n 15 .venv/bin/python convert_jpeg.py data/$name.hdf5 >> logs/gen_$name.log 2>&1 && rm -f data/$name.hdf5 && say "gen pallet: $name ready"
}
say "gen pallet start (2x2x2, 100 demos per slot)"
( one 7; one 0 ) & ( one 6; one 1 ) & ( one 5; one 2 ) & ( one 4; one 3 ) &
wait
for k in 1 2 3 4 5 6 7 8; do [ -f data/pallet_s$k.jpk.npz ] || say "gen pallet MISSING pallet_s$k"; done
say "gen pallet done"
