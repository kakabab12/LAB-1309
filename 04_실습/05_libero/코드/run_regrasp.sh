#!/usr/bin/env bash
# ⭐ 진단 확정 실험 — "교란된 자세에서 집기만 무너진다"
#
# 2026-09-23 에 좁혀진 진단
#   전환 뒤 팔은 목표 물체 **5cm 까지 다가간다** (A8→B3 중앙값 5.2cm, 10cm 안 100%).
#   그런데 B 성공은 0%. **이동은 되는데 집기가 안 된다.**
#
#   9/16 자세 민감도와 같은 이야기다:
#     손목 30도만 돌려 놓고 시작하면 물체를 집는 태스크가 0% 가 된다.
#
#   그리고 전환 시점의 이탈은 위치 27cm + 회전 8도다.
#
# 이 실험이 정하는 것
#   ① 전환 뒤 자세에서 **집기만** 시키면 되는가 (B 를 '집기'만 있는 태스크로)
#   ② 집기 실패가 **위치 때문인가 회전 때문인가** (교란을 따로 준다)
#   ③ 어느 정도까지 허용되는가 → LoRA 4차의 학습 범위를 정한다
#
# ⚠️ LoRA 2차는 8~20cm **임의 교란**으로 학습하고 **8cm 까지만** 검증했다.
#    전환 자세(27cm)는 그 범위 밖이었다. 4차는 이 결과로 범위를 정한다.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%H:%M:%S) $*"; }

if [ $# -ge 1 ]; then
  log "PID $1 대기"
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi

log "[1/2] 집는 태스크 vs 안 집는 태스크 — 위치 교란만"
for t in 1 8 9; do      # 1·8·9 = 물체를 집어야 함
  for o in 0 12 20 27; do
    $PY pose_sensitivity.py --task "$t" --offset-cm "$o" --episodes 10 \
      --save-traj --out outputs/regrasp 2>&1 | grep -E "$F"
  done
done
for t in 0 7; do        # 0·7 = 집을 필요 없음
  for o in 0 12 20 27; do
    $PY pose_sensitivity.py --task "$t" --offset-cm "$o" --episodes 10 \
      --save-traj --out outputs/regrasp 2>&1 | grep -E "$F"
  done
done

log "[2/2] 전환 시점과 같은 교란 (위치 27cm + 회전 8도)"
for t in 1 8 9 0 7; do
  $PY pose_sensitivity.py --task "$t" --offset-cm 27 --yaw-deg 8 --episodes 10 \
    --save-traj --out outputs/regrasp 2>&1 | grep -E "$F"
done

log "끝 — analyze_regrasp.py 로 분석"
