#!/usr/bin/env bash
# Beyond-range robustness conditions (outside the v6 domain-randomisation ranges), after run_post.sh's runs.
set -u
cd "$(dirname "$0")"
PY="nice -n 12 .venv/bin/python"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a logs/queue.log; }
ck() { echo "runs/$1/ckpt_030000"; }
wait_runs() { for r in "$@"; do until [ -f "$(ck $r)/model.safetensors" ]; do sleep 120; done; done; }
beyond() {  # tag r1 r2 r3
  wait_runs $2 $3 $4
  say "robust beyond-range $1 start"
  $PY robust_eval.py --ckpt $(ck $2) $(ck $3) $(ck $4) --tag $1 --trials 20 --conds very_dark very_bright > logs/robust_${1}_c.log 2>&1 &
  $PY robust_eval.py --ckpt $(ck $2) $(ck $3) $(ck $4) --tag $1 --trials 20 --conds warm_light table_checker > logs/robust_${1}_d.log 2>&1 &
  wait
  say "robust beyond-range $1 done"
}
beyond v4_n100 s1_v4_n100 s2_v4_n100 s3_v4_n100
until grep -q "robust v6c_n1000 done" logs/queue.log; do sleep 120; done  # keep the CPU for run_post.sh first
beyond v5_n1000 s1_v4_n1000 s2_v4_n1000 s3_v5_n1000
beyond v6c_n1000 s1_v6c_n1000 s2_v6c_n1000 s3_v6c_n1000
beyond v6_n1000 s1_v6_n1000 s2_v6_n1000 s3_v6_n1000
say "beyond-range robustness finished"
