#!/usr/bin/env bash
# ⭐ 정책은 지시 "내용"을 듣는가, "바뀌었다"는 것만 아는가
#
# 관찰 (지금까지)
#   전환하면 로봇이 **물체를 100% 내려놓는다.** 즉 지시가 바뀐 것을 **알아채긴 한다.**
#   그런데 B 를 못 한다 (32%).
#   → 내용을 듣는 게 아니라 **"바뀌었다"는 신호에만 반응**하는 것일 수 있다.
#
# 실험
#   전환할 때 B 대신 **뜻 없는 글자**를 넣는다. 성공 판정은 그대로 B 로 한다.
#
#     물체를 내려놓나?   내려놓으면  → 내용과 무관하게 '바뀜'에만 반응한다
#                        안 내려놓으면 → 내용을 듣고 있다 (B 라서 놓는 것)
#     A 를 계속 하나?     계속하면    → 지시를 무시하고 원래 일을 이어간다
#
# 조건
#   ① 정상 B          — 기준 (이미 있음, outputs/switch)
#   ② 뜻 없는 글자     — 알파벳 잡음
#   ③ 문법은 맞지만 무관한 지시 — "walk the dog to the park"
#   ④ A 를 그대로 다시  — '바뀌었다'는 신호조차 없는 경우
#
# ②③이 ①과 비슷하게 물체를 놓으면, 정책은 **내용이 아니라 변화에 반응**하는 것이다.
# 이건 [LIBERO-PRO](../../../02_논문노트/LIBERO-PRO.md) 가 한 실험을
# **조작 단계 안의 특정 시점**에서 하는 것이라, 그 논문이 못 본 자리다.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%H:%M:%S) $*"; }

if [ $# -ge 1 ]; then
  log "PID $1 대기"
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi

GIB="qxzf mribbel thonk vurplax"
IRR="walk the dog to the park"

for p in 8:7 4:7 1:5 2:5; do
  a="${p%:*}"; b="${p#*:}"

  log "A$a→B$b  ② 뜻 없는 글자"
  $PY switch_experiment.py --task-a "$a" --task-b "$b" --switch-at grasp:3 \
    --strategy flush --b-text "$GIB" --episodes 10 --out outputs/gib 2>&1 | grep -E "$F"

  log "A$a→B$b  ③ 문법은 맞지만 무관한 지시"
  $PY switch_experiment.py --task-a "$a" --task-b "$b" --switch-at grasp:3 \
    --strategy flush --b-text "$IRR" --episodes 10 --out outputs/gib2 2>&1 | grep -E "$F"
done

log "끝 — analyze_gibberish.py 로 분석"
