#!/usr/bin/env bash
# 11/2 1시: w11mb 가 ⓑ (w10m + RC 흔든·많음) 를 마치면 학습 전에 멈추고 (piper-holdlock 이 자물쇠를 쥐고 있어 학습을 못 띄움),
#   쥔 물체 따라가기 입력(wfeat3sch)으로 ③ 학습·④ 평가 (run_world11m_c.sh → piper-w11mc), 이어서 w12m (run_world12m.sh → piper-w12m) 을 띄운다.
cd "$(dirname "$0")"
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
until grep -qE "ⓑ 끝|멈춤" outputs/piper/world11mb.log; do sleep 5; done
grep -q "멈춤" outputs/piper/world11mb.log && { log "w11mb 멈춤 — 그만"; exit 1; }
sleep 3
systemctl --user stop piper-w11mb
log "w11mb 멈춤 (ⓑ 끝)"
systemctl --user stop piper-holdlock
systemd-run --user --unit=piper-w11mc --collect --setenv=FEAT=wfeat3sch bash -c "/home/user/smolVLA/run_world11m_c.sh > /home/user/smolVLA/outputs/piper/world11mc.log 2>&1"
log "piper-w11mc 띄움 (FEAT=wfeat3sch)"
systemd-run --user --unit=piper-w12m --collect bash -c "/home/user/smolVLA/run_world12m.sh > /home/user/smolVLA/outputs/piper/world12m.log 2>&1"
log "piper-w12m 띄움 (w11mc 끝을 기다림)"
