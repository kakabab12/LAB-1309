#!/usr/bin/env bash
# 10/31: w11m ② (사진 특징 계산, GPU 를 꽉 씀)·③ (학습 7GB) 는 자물쇠 없이 바로 시작한다 → 그 전에 RC 평가(piper-rc1031)를 멈추고,
#   ④ 가 시작되면 다시 띄운다 (평가는 끝난 장면을 건너뛰어 이어서). GIF 는 멈추지 않음 (그 스크립트는 꺼질 때 자물쇠를 안 놓음). 끄기는 systemctl --user stop piper-rcguard.
cd "$(dirname "$0")"
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
until grep -q "\[② " outputs/piper/world11m.log; do
  systemctl --user is-active --quiet piper-rc1031 || { grep -q "끝$" outputs/piper/rc1031.log && { log "RC 끝 — 지킬 것 없음"; exit 0; }; }
  sleep 20
done
for u in piper-rc1031; do
  systemctl --user is-active --quiet $u && { systemctl --user stop $u; log "② 시작 → $u 멈춤"; }
done
until grep -qE "\[④ |멈춤" outputs/piper/world11m.log; do sleep 60; done
grep -q "멈춤" outputs/piper/world11m.log && { log "w11m 이 멈춤 — RC 는 손으로"; exit 1; }
if ! grep -q "끝$" outputs/piper/rc1031.log; then
  systemd-run --user --unit piper-rc1031b --working-directory=/home/user/smolVLA /usr/bin/bash -c "/home/user/smolVLA/run_rc_1031.sh > /home/user/smolVLA/outputs/piper/rc1031b.log 2>&1"
  log "④ 시작 → RC 이어서 (piper-rc1031b)"
fi
