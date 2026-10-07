#!/usr/bin/env bash
# 색·모양 검출 결과 미리 계산 (10/6 낮): 블록 장면 학습 데이터 전부, 6프로세스
cd "$(dirname "$0")"
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
log "검출 미리 계산 시작"
for s in 0 1 2 3 4 5; do ( .venv/bin/python objfeat_cache.py --data data/piper_blk_dagger_b1_t data/piper_blk_dagger_ba1_t data/piper_blk_normal2_t data/piper_blk_normal3_t data/piper_blk_normal4_t data/piper_blk_normal_t data/piper_blk_recovery_t data/piper_blk_switch2_t data/piper_blk_switch3_t data/piper_blk_switch4_t data/piper_blk_switch_t data/piper_blk_wide_normal_t data/piper_blk_wide_recovery_t data/piper_blk_wide_switch_t  --shard $s --nshard 6 > outputs/piper/objfeat_$s.log 2>&1 ) & done
wait
log "검출 미리 계산 끝: $(ls data/piper_blk_*_t/episodes/*.objfeat.npy | wc -l) 에피소드"
