#!/usr/bin/env bash
# PiPER 3차 수집 (10/3 23:30): 그릇을 1.2cm 더 깊이 쥐는 시범 프로그램으로 그릇이 들어가는 시범만 다시 모은다.
#   PiPER 1차 학생 모델이 그릇 테두리를 0.7cm 높게 닫아 빈손 — 시범이 테두리 끝 0.2cm 만 걸쳐 쥐어 여유가 없었다.
#   정상 T1·T3·T4·T8 × 2000~2099, 전환·재개 12쌍 × 2000~2049. 그릇 없는 과제(T0·T2·T5·T6·T7·T9) 시범은 그대로 쓴다.
cd "$(dirname "$0")"
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
C="piper_sim/collect_piper.py"
log "PiPER 3차 수집 시작 (깊게 쥐기)"
( $PY $C normal --tasks 8 1 4 3 --episodes 2000-2099 --dart 0.1 --seed 3 --out data/piper_normal3 > outputs/piper/collect3_n.log 2>&1 ) &
( $PY $C switch --pairs 8:0,8:3,8:5,8:7,8:9,4:5 --episodes 2000-2049 --dart 0.1 --seed 3 --out data/piper_switch3 > outputs/piper/collect3_s1.log 2>&1 ) &
( $PY $C switch --pairs 4:9,1:7,8:1,8:4,1:8,4:1 --episodes 2000-2049 --dart 0.1 --seed 4 --out data/piper_switch3 > outputs/piper/collect3_s2.log 2>&1 ) &
wait
for d in piper_normal3 piper_switch3; do $PY piper_sim/trim_stalls.py data/$d data/${d}_t; done
# 그릇 없는 과제의 예전 시범만 모은다 (얕게 쥐던 그릇 시범은 뺀다)
mkdir -p data/piper_keep_t/episodes
for d in data/piper_normal_t data/piper_normal2_t; do
  for t in 0 2 5 6 7 9; do for f in $d/episodes/N${t}_ep*.npz; do ln -sf "$(readlink -f $f)" data/piper_keep_t/episodes/$(basename $(dirname $(dirname $f)))_$(basename $f); done; done
done
log "PiPER 3차 수집 끝: 정상 $(ls data/piper_normal3/episodes | wc -l), 전환 $(ls data/piper_switch3/episodes | wc -l), 예전 그대로 $(ls data/piper_keep_t/episodes | wc -l)"
echo "data/piper_keep_t data/piper_normal3_t data/piper_switch3_t" > outputs/piper/train_data.txt
