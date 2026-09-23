#!/usr/bin/env bash
# 2026-09-23 발견을 **눈으로 보여주는** 영상
#
# 오늘 확정된 것: 교란된 자세에서 무너지는 것은 '이동'이 아니라 '집기' 다.
#   교란 27cm 에서  서랍 열기 62%  vs  물체 집기 4%
#
# 보여줄 것
#   ① 잘 되는 전환   A8→B0 (그릇 들다가 "서랍 열어라")     — 60% 성공
#   ② 안 되는 전환   A8→B3 (그릇 들다가 "그릇을 서랍에")   — 0%, 놓고 안 돌아감
#   ③ 같은 교란, 다른 태스크: 팔을 27cm 옮겨 놓고 시작
#        서랍 열기(T0)  vs  그릇 집기(T8)
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
F='SUMMARY|Traceback|Error'
log(){ echo "=== $(date +%H:%M:%S) $*"; }

if [ $# -ge 1 ]; then
  log "PID $1 대기"
  while kill -0 "$1" 2>/dev/null; do sleep 30; done
fi

log "[1/3] 잘 되는 전환 — 그릇 들다가 '서랍 열어라' (A8→B0)"
$PY switch_experiment.py --task-a 8 --task-b 0 --switch-at grasp:3 --strategy flush \
  --episodes 4 --video --out outputs/media_ok 2>&1 | grep -E "$F"

log "[2/3] 안 되는 전환 — 그릇 들다가 '그릇을 서랍에' (A8→B3)"
$PY switch_experiment.py --task-a 8 --task-b 3 --switch-at grasp:3 --strategy flush \
  --episodes 4 --video --out outputs/media_bad 2>&1 | grep -E "$F"

log "[3/3] 같은 교란(27cm)에서 시작 — 서랍 열기 vs 그릇 집기"
$PY pose_sensitivity.py --task 0 --offset-cm 27 --episodes 3 --video --out outputs/media_pose0 2>&1 | grep -E "$F"
$PY pose_sensitivity.py --task 8 --offset-cm 27 --episodes 3 --video --out outputs/media_pose8 2>&1 | grep -E "$F"

log "끝 — make_gifs.py 로 GIF 만들 것"
