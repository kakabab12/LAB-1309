#!/usr/bin/env bash
# 200 more demonstrations (4 x 50, new seeds) for the weaker pallet slots 2, 3, 4, 5 and 8 (0-based 1, 2, 3, 4, 7),
# made ahead of time while the slot-7 300-demo test (pal_s7b) runs; 6 generator processes at a time.
set -u
cd "$(dirname "$0")"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> logs/queue.log; }
one() {  # slot(0-based) part
  local k=$1 p=$2 name=pallet_s$(($1 + 1))_m$2
  [ -f data/$name.jpk.npz ] && return 0
  [ -f data/${name}_gen.json ] || nice -n 15 .venv/bin/python gen_pallet.py --slot $k --episodes 50 --seed $((41000 + 1000 * p)) \
      --name $name --out data > logs/gen_$name.log 2>&1
  nice -n 15 .venv/bin/python convert_jpeg.py data/$name.hdf5 >> logs/gen_$name.log 2>&1 && rm -f data/$name.hdf5
}
export -f one
say "gen pallet more: +200 demos for slots 2,3,4,5,8 (20 parts x 50, 6 at a time)"
for k in 1 2 3 4 7; do for p in 0 1 2 3; do echo "$k $p"; done; done | xargs -P 6 -L 1 bash -c 'one $0 $1'
miss=""; for k in 2 3 4 5 8; do for p in 0 1 2 3; do [ -f data/pallet_s${k}_m$p.jpk.npz ] || miss="$miss s${k}_m$p"; done; done
say "gen pallet more done${miss:+ (MISSING$miss)}"
