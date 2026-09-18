#!/usr/bin/env bash
# 주말(9/19~20) 동안 GPU 가 놀지 않게. CEM 결과에 따라 갈라진다.
# 사람은 쉬고 연구실 PC 만 돌린다 — 월요일(9/21) 아침에 결과를 분석한다.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
W=${1:-0}
log(){ echo "=== $(date +%H:%M:%S) $*"; }
F='SUMMARY|Traceback|Error'
if [ "$W" != "0" ]; then
  log "PID $W 대기"
  while kill -0 "$W" 2>/dev/null; do sleep 120; done
fi

log "CEM 결과 판정"
$PY analyze_seedtest.py outputs/cemtest 2>&1 | tail -12
VERDICT=$($PY - <<'PY'
import json, os
# cemtest: 기본 vs noise-shift 비교. shift 쪽이 10%p 이상 높으면 성공으로 본다
import glob
base=shift=None; bn=sn=0
for f in glob.glob("outputs/cemtest/A*_B*_grasp3_flush*.json"):
    d=json.load(open(f)); a=d["args"]
    sw=[e for e in d["episodes"] if e.get("switched")]
    if not sw: continue
    k=sum(bool(e.get("b_success")) for e in sw)
    if a.get("noise_shift"):
        shift=(shift or 0)+k; sn+=len(sw)
    else:
        base=(base or 0)+k; bn+=len(sw)
if base is None or shift is None or not bn or not sn:
    print("unknown")
else:
    print("cem_works" if shift/sn - base/bn >= 0.10 else "cem_fails")
PY
)
log "판정: $VERDICT"

case "$VERDICT" in
  cem_works)
    log "[주말 1] CEM 을 4→7 에도 (쌍마다 찾아야 하는지 확인)"
    $PY cem_noise.py --task-a 4 --task-b 7 --iters 8 --candidates 6 --episodes 6 \
      --out outputs/cem_4_7 2>&1 | tail -20
    if [ -f outputs/cem_4_7/mu_best.npy ]; then
      for s in "" "--noise-shift outputs/cem_4_7/mu_best.npy"; do
        $PY switch_experiment.py --task-a 4 --task-b 7 --switch-at grasp:3 --strategy flush \
          $s --episodes 10 --start-episode 10 --out outputs/cemtest 2>&1 | grep -E "$F"
      done
    fi
    log "[주말 2] 8→7 μ 를 더 많은 에피소드로 검증 (통계 강화)"
    for s in "" "--noise-shift outputs/cem_8_7/mu_best.npy"; do
      $PY switch_experiment.py --task-a 8 --task-b 7 --switch-at grasp:3 --strategy flush \
        $s --episodes 20 --start-episode 20 --out outputs/cemtest20 2>&1 | grep -E "$F"
    done
    log "[주말 3] 한 쌍의 μ 를 다른 쌍에 써 보기 (일반화 확인)"
    $PY switch_experiment.py --task-a 6 --task-b 0 --switch-at grasp:3 --strategy flush \
      --noise-shift outputs/cem_8_7/mu_best.npy --episodes 10 --start-episode 10 \
      --out outputs/cemcross 2>&1 | grep -E "$F"
    ;;
  *)
    log "[주말 1] CEM 이 안 통했다 → 진단: 조종 가능성 CMI 를 더 넓게"
    $PY steerability.py --tasks 8 4 6 9 --episodes 4 --k 8 --out outputs/steer_wide 2>&1 | tail -20
    $PY analyze_steer.py 2>&1 | tail -15
    log "[주말 2] 노이즈 시드 스윕 16개 (더 좋은 노이즈가 아예 있는지)"
    $PY oracle_bon.py --pairs 8:7,6:0 --episodes 8 --n 16 --out outputs/oracle_n16 2>&1 | tail -20
    $PY analyze_oracle.py outputs/oracle_n16 2>&1 | tail -20
    ;;
esac
log "주말 큐 끝 — 월요일 아침에 분석"
