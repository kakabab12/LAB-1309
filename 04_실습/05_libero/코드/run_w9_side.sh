#!/usr/bin/env bash
# 10/30 새벽: piper-w9 가 w9 일 1개 평가(다듬기·혼자)를 하는 동안만 GPU 4번째 자리에서 w9 + 다듬기 새 부품 전부 (일 1개) 를 돌린다.
#   새 부품: fmask(손가락 지우기)·settle(닫힐 때까지 붙잡기)·wxy(손목 사진으로 자리 고치기)·gplan(쥐는 자세 계산)·objc(든 물체 가운데로 놓기)·pfine(놓기 마무리)
#   → outputs/world_w9ra_{설정}/k1n.json (w9r 과 같은 장면 비교). 파이프라인이 일 2개·3개 평가로 넘어가면 (지시문이 여러 개라 메모리가 커짐)
#   PID 로 끄고 그만둔다. 남은 장면은 run_world10.sh ⓪ 이 마저 잰다 (world_eval 은 끝난 장면을 건너뜀).
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
L=outputs/piper/w9_side.log
P9=outputs/piper/world9.log
log(){ echo "=== $(date +%m/%d\ %H:%M) $*" >> $L; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
RA="grasp,place,fmask,settle,wxy,gplan,objc,pfine"
phase_k1(){ grep -q '\[⑤ w9\] 일 1개' $P9 && ! grep -q '\[⑤ w9\] 일 2개' $P9; }
done_all(){ grep -q '^=== .* 끝$' $P9 && grep -q '\[⑤ w9\]' $P9; }
log "기다림: w9 일 1개 평가 시작"
until phase_k1; do
  systemctl --user is-active --quiet piper-w9 || { log "piper-w9 멈춤·끝 — 그만"; exit 0; }
  sleep 60
done
sleep 240
for c in default many random; do
  while true; do
    if done_all; then log "파이프라인 끝 — 그만"; exit 0; fi
    if phase_k1; then break; fi
    sleep 60                                           # 일 2·3개 평가 중에는 기다림
  done
  f=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F, '{print $1-$2}')
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  if [ "$f" -lt 2900 ] || [ "$n" -ge 4 ]; then log "$c: GPU 남은 ${f}MiB·프로세스 $n — 1분 뒤 다시"; sleep 60; continue; fi
  PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_eval.py --policy outputs/piper_w9_model/merged --seeds 1000-1049 --k 1 --nested $EVQ \
      --refine $RA --out outputs/world_w9ra_$c/k1n.json > outputs/piper/w9_side_$c.log 2>&1 &
  P=$!; log "$c 띄움 (PID $P, GPU 남은 ${f}MiB)"
  while kill -0 $P 2>/dev/null; do
    if ! phase_k1; then kill -TERM $P; sleep 3; kill -KILL $P 2>/dev/null; log "$c: 파이프라인이 일 2개 평가로 넘어감 — 끔 (남은 장면은 run_world10.sh ⓪)"; break; fi
    sleep 20
  done
  grep -q SUMMARY outputs/piper/w9_side_$c.log && grep SUMMARY outputs/piper/w9_side_$c.log | tail -1 | sed "s|^|w9ra_$c |" >> $L
  if ! phase_k1; then log "$c: 일 1개 평가 단계가 끝남 — 그만 (남은 장면은 run_world10.sh ⓪ 이 마저 잰다)"; break; fi
done
log "끝"
