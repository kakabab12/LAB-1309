#!/usr/bin/env bash
# 800 more DART demos per stage (-> DART 1000 = dart_stage{s} 200 + dart_stage{s}_more 800), one at a time.
cd "$(dirname "$0")"
PYLOW="nice -n 15 .venv/bin/python"
for s in 1 2 3; do
  stem=dart_stage${s}_more
  [ -f data/${stem}.jpk.npz ] && continue
  [ -f data/${stem}_gen.json ] || $PYLOW gen_data.py --stage $s --episodes 800 --seed 7000 --dart-sigma 0.005 \
      --name $stem --out data > logs/gen_${stem}.log 2>&1
  $PYLOW convert_jpeg.py data/${stem}.hdf5 >> logs/gen_${stem}.log 2>&1
done
echo done > logs/dart_more_data.done
echo "[$(date '+%m-%d %H:%M:%S')] DART 800 more demos per stage ready" >> logs/queue.log
