#!/usr/bin/env bash
# 10/29 밤: piper-w9 가 학습 데이터 위치 입력 다시 계산(CPU)을 기다리는 동안 GPU 가 빈다 (DAgger 6차·사진 특징 뒤 ~ 02시쯤).
#   그 사이에만 w8 로 시험을 돌리고, 위치 입력 계산이 끝나면 (학습 시작 전) PID 로 끈다. world_eval 은 끝난 장면을 건너뛰므로
#   끊겨도 그때까지 잰 장면은 남는다.
#   ① w8 + 다듬기 + gplan (쥐는 자세 계산, world_gplan), 일 1개, 기본·물체 많음 1000~1049 → w8rm 과 같은 장면 비교
#      (objc 는 다음 빈 때로 미룸)
#   ② 깊이 3 끼어들기 DAgger 시험 (world_dagger_deep, 시범 프로그램이 다시 못 쥐는 까닭 WE.why 보기)
#   ③ 막대 쥐기 실패 12장면 + 판 근처에 놓은 실패 14장면 따라가기 (diag_bar.py: 오므린·편 순간의 높이·어긋남, 다듬기 상태) — 한 자리에서 차례로
#   GPU 는 한꺼번에 3개만 (DAgger 처럼 지시문 길이가 여러 가지면 하나가 2.9GB 까지 커진다): ① 둘 + ③ 다음 ②를 한 자리에서 차례로
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
L=outputs/piper/idle_w9.log
log(){ echo "=== $(date +%m/%d\ %H:%M) $*" >> $L; }
stop_now(){ grep -q '\[②\] 끝' outputs/piper/world9_feat.log 2>/dev/null || grep -q '\[④' outputs/piper/world9.log; }
gpu_free(){ nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F, '{print $1-$2}'; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
C8=outputs/piper_w8_model/merged

log "기다림: piper-w9 가 '[②] 끝나기 기다림' 에 닿을 때까지"
until grep -q '\[②\] 끝나기 기다림' outputs/piper/world9.log; do
  systemctl --user is-active --quiet piper-w9 || { log "piper-w9 가 멈춰 있음 — 안 띄움"; exit 0; }
  sleep 30
done
sleep 30
stop_now && { log "위치 입력 계산이 이미 끝남 — 안 띄움"; exit 0; }
PIDS=()
launch(){   # $1 이름, 나머지 명령. GPU 남은 메모리 3.5GB 넘을 때만
  local name=$1; shift
  local f; f=$(gpu_free)
  [ "$f" -gt 3500 ] || { log "$name: GPU 남은 메모리 ${f}MiB — 안 띄움"; return; }
  stop_now && return
  "$@" > outputs/piper/idle_$name.log 2>&1 &
  PIDS+=($!); log "$name 띄움 (PID $!, GPU 남은 ${f}MiB)"; sleep 60
}
for c in default many; do
  launch w8rmg_$c env PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_eval.py --policy $C8 --seeds 1000-1049 --k 1 --nested $EVQ \
      --refine grasp,place,gplan --out outputs/world_w8rmg_$c/k1n.json
done
launch diag bash -c "PIPER_WORLD=piper_sim/worlds/default.json $PY piper_sim/diag_bar.py --policy $C8 --seeds 1004,1022,1024,1036 --out outputs/piper/diag_bar_default.json $EVQ; \
    PIPER_WORLD=piper_sim/worlds/random.json $PY piper_sim/diag_bar.py --policy $C8 --seeds 1009,1015,1022 --out outputs/piper/diag_bar_random.json $EVQ; \
    PIPER_WORLD=piper_sim/worlds/many.json $PY piper_sim/diag_bar.py --policy $C8 --seeds 1008,1027,1038,1041,1028 --out outputs/piper/diag_bar_many.json $EVQ; \
    PIPER_WORLD=piper_sim/worlds/default.json $PY piper_sim/diag_bar.py --policy $C8 --seeds 1009,1039,1047,1018,1046 --out outputs/piper/diag_place_default.json $EVQ; \
    PIPER_WORLD=piper_sim/worlds/random.json $PY piper_sim/diag_bar.py --policy $C8 --seeds 1014,1028,1030,1032,1013,1039 --out outputs/piper/diag_place_random.json $EVQ; \
    PIPER_WORLD=piper_sim/worlds/many.json $PY piper_sim/diag_bar.py --policy $C8 --seeds 1003,1021,1048 --out outputs/piper/diag_place_many.json $EVQ; \
    PIPER_WORLD=piper_sim/worlds/default.json $PY piper_sim/world_dagger_deep.py --policy $C8 --depth 3 --seeds 13000-13019 --out data/world_daggerK3_test $EVQ"

while true; do
  stop_now && { log "위치 입력 계산 끝 — 학습 전에 끔"; break; }
  alive=0; for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
  [ $alive -eq 0 ] && { log "모두 끝남"; break; }
  sleep 10
done
for p in "${PIDS[@]}"; do kill -STOP "$p" 2>/dev/null; pkill -TERM -P "$p" 2>/dev/null; done   # 묶음(bash -c)은 먼저 멈춰 다음 것을 못 띄우게 하고 자식 python 까지 (부모 PID 로)
sleep 3
for p in "${PIDS[@]}"; do pkill -KILL -P "$p" 2>/dev/null; kill -KILL "$p" 2>/dev/null; done
for c in default many; do grep -h SUMMARY outputs/piper/idle_w8rmg_$c.log 2>/dev/null | tail -1 | sed "s|^|w8rmg_$c |" >> $L; done
grep -h -E "^== |STATS" outputs/piper/idle_diag.log 2>/dev/null | cut -c1-200 >> $L
log "끝"
