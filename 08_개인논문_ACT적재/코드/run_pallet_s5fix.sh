#!/usr/bin/env bash
# Pallet slot 5 fix. Slot 5 (upper layer, far corner) stayed at 2-9/20 whatever the state noise: the expert itself was
# 6.2 mm off there (other slots 1.8 mm) because the IK gave up 7-11 mm of position for the approach tilt at the reach
# limit. PALLET_IK_POS=1 re-solves such waypoints position-first (pallet_grid.GridExpert._ik): expert 0-3.5 mm.
# New slot-5 demos pallet3_s5 (4 x 25) -> pal3n05_s5 (state noise 0.05, like the other v3 slots) -> slot check (20,
# same protocol as slotcheck_pal2n*_s5) -> full 2x2x2 eval (2 x 20 sequences, k 25) with pal3n05_s5 if it is the best
# slot-5 policy, once the v3 slot policies (orchestrator) are trained.
set -u
cd "$(dirname "$0")"
export PALLET_X0=0.20 PALLET_Y0=0.017 PALLET_GAP_Y=0.014 PALLET_VIA=far ACT_DESIGN=v4
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> logs/queue.log; }
n_train() { pgrep -fc "^.venv/bin/python train_act.py" || true; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
ck() { [ -f "runs/$1/ckpt_030000/model.safetensors" ]; }

# 1) demonstrations
if [ ! -f data/pallet3_s5.jpk.npz ]; then
  say "pallet slot 5 fix: position-first IK expert, 4 x 25 demos"
  for p in 0 1 2 3; do
    [ -f "data/pallet3_s5_p${p}_gen.json" ] || PALLET_IK_POS=1 nice -n 15 .venv/bin/python gen_pallet.py --slot 4 \
        --episodes 25 --seed $((63000 + p)) --name "pallet3_s5_p$p" --out data > "logs/gen_pallet3_s5_p$p.log" 2>&1 &
    sleep 10
  done
  wait
  nice -n 15 .venv/bin/python convert_jpeg.py data/pallet3_s5_p0.hdf5 data/pallet3_s5_p1.hdf5 data/pallet3_s5_p2.hdf5 \
      data/pallet3_s5_p3.hdf5 --out data/pallet3_s5.jpk.npz > logs/gen_pallet3_s5.log 2>&1 \
    && rm -f data/pallet3_s5_p0.hdf5 data/pallet3_s5_p1.hdf5 data/pallet3_s5_p2.hdf5 data/pallet3_s5_p3.hdf5
fi
[ -f data/pallet3_s5.jpk.npz ] || { say "pallet slot 5 fix: data/pallet3_s5.jpk.npz MISSING"; exit 1; }
say "pallet slot 5 fix: demos ready ($(python3 -c "
import json
k = t = 0; e = []
for p in range(4):
    s = json.load(open(f'data/pallet3_s5_p{p}_gen.json'))
    k += s['summary']['kept']; t += s['summary']['tried']; e += [r['xy_err_mm'] for r in s['log'] if r['success']]
print(f'{k}/{t}, expert xy err mean {sum(e) / len(e):.1f} mm, max {max(e):.1f} mm (before: 6.2 / 11.5)')"))"

# 2) training (GPU limits shared with the orchestrator, which counts this training too)
if ! ck pal3n05_s5; then
  while [ "$(n_train)" -ge 4 ] || [ "$(gpu_free)" -lt 3000 ]; do sleep 60; done
  say "start train pal3n05_s5 (slot 5, position-first expert demos, state noise 0.05)"
  nice -n 10 .venv/bin/python train_act.py --stage 5 --episodes 100 --steps 30000 --save-every 10000 \
      --data data/pallet3_s5.jpk.npz --out runs/pal3n05_s5 --state-noise 0.05 >> logs/train_pal3n05_s5.log 2>&1
fi
ck pal3n05_s5 || { say "pallet slot 5 fix: pal3n05_s5 not trained"; exit 1; }

# 3) slot check (same protocol as the other slot-5 checks)
if [ ! -f results/pallet/slotcheck_pal3n05_s5.json ]; then
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 .venv/bin/python pallet_slot_check.py --slot 4 --run pal3n05_s5 \
      --trials 20 --k 25 --out results/pallet/slotcheck_pal3n05_s5.json > logs/slotcheck_pal3n05_s5.log 2>&1
fi
best=$(python3 -c "
import json
r = {n: sum(x['success'] for x in json.load(open(f'results/pallet/slotcheck_{n}.json'))['rows'])
     for n in ('pal2n05_s5', 'pal2n10_s5', 'pal3n05_s5')}
print(max(r, key=lambda n: (r[n], n == 'pal3n05_s5')), ' '.join(f'{n} {v}/20' for n, v in r.items()))")
say "pallet slot 5 fix: slot check $(echo "$best" | cut -d' ' -f2-) -> best $(echo "$best" | cut -d' ' -f1)"
[ "$(echo "$best" | cut -d' ' -f1)" = pal3n05_s5 ] || { say "pallet slot 5 fix: no better slot-5 policy, full eval skipped"; exit 0; }

# 4) full 2x2x2 sequences with the fixed slot 5
for k in 1 2 3 4 6 7 8; do until ck "pal2n05_s$k"; do sleep 300; done; done
say "pallet slot 5 fix: full 2x2x2 eval with pal3n05_s5 (2 x 20 sequences)"
for p in 0 1; do
  [ -f "results/eval/pallet4_k25_p$p/results.json" ] && continue
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 .venv/bin/python pallet_eval.py --runs pal2n05_s1 pal2n05_s2 \
      pal2n05_s3 pal2n05_s4 pal3n05_s5 pal2n05_s6 pal2n05_s7 pal2n05_s8 --k 25 --start $((p * 20)) --trials 20 \
      --gifs $([ $p = 0 ] && echo 2 || echo 0) --out "results/eval/pallet4_k25_p$p" > "logs/eval_pallet4_p$p.log" 2>&1 &
done
wait
say "pallet slot 5 fix: done eval pallet4 ($(for p in 0 1; do grep -h "full" "logs/eval_pallet4_p$p.log" | tail -1; done | tr '\n' ' '))"
