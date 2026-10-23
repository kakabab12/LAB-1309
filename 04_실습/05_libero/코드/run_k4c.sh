#!/usr/bin/env bash
# 4K 시연 (10/24): 셋 다 90.4% 설정 (np3 + 고친 보조 전부) 으로 노란 막대 끼어들기. 28항목 나머지(piper-np3d3)가 끝나면.
#   A1→B7 장면 1019: np3 + 새 설정으로는 돌아오기 실패 (빨강을 든 채 맴돎), + ㉓ 데려다 놓기로 성공한 장면. 실패하면 다음 장면
cd "$(dirname "$0")"
export PIPER_NOPED=1 PIPER_BLOCKS=1 PIPER_GRASP_STYLE=align MUJOCO_GL=egl
while systemctl --user is-active -q piper-np3d3; do sleep 60; done
for e in 1019 1005 1010; do
  until mkdir outputs/piper/gpu_launch.lock 2>/dev/null; do sleep 10; done
  ( sleep 90; rmdir outputs/piper/gpu_launch.lock ) &
  echo "=== 장면 $e"
  timeout 2400 .venv/bin/python korean_live.py --episode $e --script "0=빨간 블록을 주황 판에 올려줘" "grasp+30=잠깐, 노란 막대 먼저 흰 판에 놓고 하던 거 이어서 해줘" \
    --render4k outputs/media/k4_all3_$e --title "PiPER 시뮬 · 한국어 끼어들기 · 셋 다 90% 설정 (np3 + 고친 보조)" 2>&1 | grep -E "^\[|끝:" 
  wait
  [ $(awk -v s="=== 장면 $e" '$0==s{f=1;next} /^===/{f=0} f' outputs/piper/k4c.log | grep -c "끝: .*성공") -ge 2 ] && { echo "성공 장면 $e"; break; }
  : > /dev/null
done
echo 끝
