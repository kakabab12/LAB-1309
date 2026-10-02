#!/usr/bin/env bash
# p1·p2(단독 태스크 DAgger)가 끝나면 p3(전환 12쌍)의 남은 쌍을 뒤에서부터 p4 가 나눠 맡는다.
# p3 가 p4 가 이미 시작한 쌍에 닿으면 p3 를 끝낸다. (GPU 프로세스 4개를 넘기지 않는다)
cd "$(dirname "$0")"
PY=.venv/bin/python
L=(8:0 8:3 8:5 8:7 8:9 4:5 4:9 1:7 8:1 8:4 1:8 4:1)
n=${#L[@]}
alive(){ ps -eo pid,comm,args | awk -v s="$1" '$2=="python" && $0 ~ s && /dagger_v6\.py/ {print $1}'; }
while [ -n "$(alive 'seed 1')" ] || [ -n "$(alive 'seed 2')" ]; do sleep 60; done
P3=$(alive 'seed 3'); [ -z "$P3" ] && { echo "p3 이미 끝"; exit 0; }
done3=$(grep -c "^  == A" outputs/v6/dagger/p3.log)
rest=$(( n - done3 - 1 ))          # p3 가 지금 하는 쌍 뒤로 남은 수
[ $rest -lt 2 ] && { echo "남은 쌍 적음 — 나누지 않음"; exit 0; }
half=$(( rest / 2 ))
R=""; for ((k=n-1; k>=n-half; k--)); do R="$R,${L[$k]}"; done; R=${R#,}
echo "$(date +%H:%M) p3 완료 $done3, p4 가 맡을 쌍: $R"
setsid $PY dagger_v6.py --policy outputs/v6b_model/merged --latency-steps 11 --episodes 0-19,50-69 \
  --out data/v6_dagger_lat --pairs $R --seed 4 > outputs/v6/dagger/p4.log 2>&1 &
while kill -0 $P3 2>/dev/null; do
  i3=$(grep -c "^  == A" outputs/v6/dagger/p3.log)      # p3 가 지금 하는 쌍 번호
  j=$(grep -c "^  == A" outputs/v6/dagger/p4.log)       # p4 가 끝낸 수
  last4=$(( j < half ? j : half - 1 ))
  if [ $i3 -ge $(( n - 1 - last4 )) ]; then echo "$(date +%H:%M) p3 가 p4 영역(${L[$i3]})에 닿음 — p3 종료"; kill $P3; break; fi
  sleep 30
done
wait
echo "$(date +%H:%M) 나눠 수집 끝"
