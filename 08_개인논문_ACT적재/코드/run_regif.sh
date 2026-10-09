#!/usr/bin/env bash
cd /home/user/ACT/act_sim
V5="runs/s1_v4_n100/ckpt_030000 runs/s2_v4_n100/ckpt_030000 runs/s3_v5_n100/ckpt_030000"
V6="runs/s1_v6_n1000/ckpt_030000 runs/s2_v6_n1000/ckpt_030000 runs/s3_v6_n1000/ckpt_030000"
P="nice -n 10 .venv/bin/python clutter_eval.py --mode chained --trials 2 --gifs 2 --factory-env --out results/clutter_gifs"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 $P --ckpt $V6 --tag v6_n1000 --conds all tight factory > logs/regif_v6a.log 2>&1 &
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 $P --ckpt $V6 --tag v6_n1000 --conds factory_vis factory_max > logs/regif_v6b.log 2>&1 &
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 $P --ckpt $V5 --tag v5_n100 --conds factory_max > logs/regif_v5.log 2>&1 &
wait
echo "[$(date '+%m-%d %H:%M:%S')] regif done" >> logs/queue.log
