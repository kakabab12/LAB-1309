#!/usr/bin/env bash
# Post-training evaluations on the CPU (policy on CPU, 2 processes per task): robustness conditions and
# scale-verified retry, each task waiting until its checkpoints exist. Runs next to the GPU training queues.
# Launch:  systemd-run --user --unit=act-post -p MemoryMax=6G -p Nice=12 -p WorkingDirectory=$PWD \
#            --setenv=MUJOCO_GL=egl --setenv=CUDA_VISIBLE_DEVICES= --setenv=OMP_NUM_THREADS=2 \
#            --setenv=MALLOC_MMAP_THRESHOLD_=1048576 /bin/bash $PWD/run_post.sh
set -u
cd "$(dirname "$0")"
PY="nice -n 12 .venv/bin/python"
LOG=logs/queue.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
ck() { echo "runs/$1/ckpt_030000"; }
wait_runs() { for r in "$@"; do until [ -f "$(ck $r)/model.safetensors" ]; do sleep 120; done; done; }
ALL_A="base pos_out yaw_out placed_out dark bright light_side table_gray"
ALL_B="table_dark cam_top cam_wrist distractor mass_100g mass_200g delay_67ms delay_133ms"
LIGHT="base dark bright light_side"

robust() {  # tag r1 r2 r3 "conds A" "conds B"
  local tag=$1 r1=$2 r2=$3 r3=$4 ca=$5 cb=$6
  wait_runs $r1 $r2 $r3
  say "robust $tag start"
  for part in "a|$ca" "b|$cb"; do
    local n=${part%%|*} conds=${part#*|}
    [ -n "$conds" ] && $PY robust_eval.py --ckpt $(ck $r1) $(ck $r2) $(ck $r3) --tag $tag --trials 20 --conds $conds \
        > logs/robust_${tag}_$n.log 2>&1 &
  done
  wait
  say "robust $tag done"
}

retry() {  # name r1 r2 r3
  local name=$1 r1=$2 r2=$3 r3=$4
  [ -f results/eval/${name}_retry/results.json ] && return
  wait_runs $r1 $r2 $r3
  say "retry $name start"
  $PY eval_retry.py --ckpt $(ck $r1) $(ck $r2) $(ck $r3) --out results/eval/${name}_retry --trials 50 > logs/retry_${name}.log 2>&1
  say "retry $name done: $(grep '\[retry\]' logs/retry_${name}.log)"
}

say "post-eval queue start"
# v4_n100: the first run (units act-robust-a/b) has the 12 original conditions; add the weight / delay ones
while systemctl --user is-active --quiet act-robust-a || systemctl --user is-active --quiet act-robust-b; do sleep 120; done
robust v4_n100 s1_v4_n100 s2_v4_n100 s3_v4_n100 "mass_100g mass_200g" "delay_67ms delay_133ms"
robust v5_n1000 s1_v4_n1000 s2_v4_n1000 s3_v5_n1000 "$ALL_A" "$ALL_B"
robust v6c_n1000 s1_v6c_n1000 s2_v6c_n1000 s3_v6c_n1000 "$ALL_A" "$ALL_B"
retry v5_n1000 s1_v4_n1000 s2_v4_n1000 s3_v5_n1000
robust v6_n1000 s1_v6_n1000 s2_v6_n1000 s3_v6_n1000 "$ALL_A" "$ALL_B"
robust v5_n200 s1_v4_n200 s2_v4_n200 s3_v5_n200 "$LIGHT" ""
robust v5c_n200 s1_v4c_n200 s2_v4c_n200 s3_v5c_n200 "$LIGHT" ""
say "post-eval queue finished"
