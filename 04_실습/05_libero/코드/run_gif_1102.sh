#!/usr/bin/env bash
# 11/2 일지용 4K GIF: w10m + RC 다듬기 기본 쌓기 성공 장면 (1015 → 안 되면 1010 → 1013). 01:12~ piper-holdlock 이 GPU 자물쇠를 쥐고 있어
#   자물쇠 없이 GPU 3개 아래만 확인하고 띄운다 (끝나면 저절로 끝). 끄기는 systemctl --user stop piper-gif1102.
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
gpu_n(){ nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -c python; }
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
for s in 1015 1010 1013; do
  while [ "$(gpu_n)" -ge 3 ]; do sleep 20; done
  log "띄움 4K default $s"
  PIPER_WORLD=piper_sim/worlds/default.json $PY piper_sim/world_vla4k.py --policy outputs/piper_w10m_model/merged --seed $s --k 1 --nested \
      --name world_w10mrc_default_s$s --label "w10m + 다듬기" --latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph \
      --refine $(cat outputs/piper/ra_choice.txt) > outputs/piper/gif1102_$s.log 2>&1
  tail -1 outputs/piper/gif1102_$s.log
  grep -q "^성공" outputs/piper/gif1102_$s.log && { log "끝 (성공 장면 default $s)"; exit 0; }
done
log "끝 (세 장면 모두 실패)"
