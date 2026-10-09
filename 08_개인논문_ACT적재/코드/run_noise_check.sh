#!/usr/bin/env bash
# Does state-input noise (train_act.py --state-noise) also help the main stacking task, and more noise the pallet?
#  - s2_v4n05_n100: stage 2 (beside), same 100 v4 demos as s2_v4_n100, noise 0.05 rad -> per-stage + chained eval
#    with s1_v4_n100 / s3_v5_n100 (compare with v5_n100: stage 2 placed +4 mm towards +y)
#  - pal2n10_s5: pallet slot 5 with 0.1 rad (0.05 rad: 2/20 -> 9/20)
set -u
cd "$(dirname "$0")"
PY="nice -n 10 .venv/bin/python"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> logs/queue.log; }
say "noise check: s2_v4n05_n100 (main task stage 2, 0.05 rad), pal2n10_s5 (0.1 rad)"
D2=$(python3 -c "import json; print(' '.join(json.load(open('runs/s2_v4_n100/train_meta.json'))['data']))")
$PY train_act.py --stage 2 --episodes 100 --steps 30000 --save-every 10000 --data $D2 --state-noise 0.05 \
    --out runs/s2_v4n05_n100 >> logs/train_s2_v4n05_n100.log 2>&1 &
sleep 60
$PY train_act.py --stage 5 --episodes 100 --steps 30000 --save-every 10000 --data data/pallet2_s5.jpk.npz --state-noise 0.1 \
    --out runs/pal2n10_s5 >> logs/train_pal2n10_s5.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 $PY eval_act.py --device cpu --ckpt runs/s1_v4_n100/ckpt_030000 \
    runs/s2_v4n05_n100/ckpt_030000 runs/s3_v5_n100/ckpt_030000 --trials 50 --gifs 1 --out results/eval/v5n05_n100 \
    > logs/eval_v5n05_n100.log 2>&1 &
PALLET_X0=0.20 PALLET_Y0=0.017 PALLET_GAP_Y=0.014 PALLET_VIA=far ACT_DESIGN=v4 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 \
    $PY pallet_slot_check.py --slot 4 --run pal2n10_s5 --trials 20 --k 25 --out results/pallet/slotcheck_pal2n10_s5.json \
    > logs/slotcheck_pal2n10_s5.log 2>&1 &
wait
say "noise check: main stage-2 noise $(grep '^\[' logs/eval_v5n05_n100.log | tr '\n' ' ') | pallet slot 5 0.1 rad $(grep '^\[slot' logs/slotcheck_pal2n10_s5.log)"
