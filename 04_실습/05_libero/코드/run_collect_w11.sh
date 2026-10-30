#!/usr/bin/env bash
# w11 시범 더 모으기 (10/30 밤): 새 기준(판 N개·동작 변수) 시범을 세 설정 × 80장면, 쌓기 몫 50% (PIPER_STACK_P — 쌓기가 가장 약함, 학습 때 obj_frac 으로 몫을 다시 맞춤).
#   w10m 학습이 끝나 ④ 평가가 시작되면 띄운다 (CPU, nice 10 — 평가가 먼저). MuJoCo 그림 그리기로 GPU 0.25GB씩.
#   끄기는 systemctl --user stop piper-co11 으로만.
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl PIPER_STACK_P=0.5
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
until grep -qE "④|멈춤" outputs/piper/world10m.log; do sleep 60; done
grep -q "멈춤" outputs/piper/world10m.log && { log "파이프라인 멈춤 — 그만"; exit 1; }
sleep 600                                            # 평가 일꾼들이 먼저 자리 잡게
P=()
for c in default:24000-24079 random:24200-24279 many:24400-24479; do
  IFS=: read -r w s <<< "$c"
  log "띄움 시범 $w $s (메모리 $(awk '/MemAvailable/{print int($2/1e6)}' /proc/meminfo)GB)"
  ( PIPER_WORLD=piper_sim/worlds/$w.json nice -n 10 $PY piper_sim/world_collect.py --seeds $s --k 1 4 --out data/world_motion2_$w \
      > outputs/piper/co11_$w.log 2>&1 ) & P+=($!)
  sleep 30
done
wait "${P[@]}"
grep -h STATS outputs/piper/co11_*.log
log "끝"
