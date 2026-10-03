#!/usr/bin/env bash
# PiPER 1차 학습이 끝나는 순간 지금 도는 Panda 기록 평가를 멈추고, 'PiPER 학습 중에만' 도는 판으로 바꾼다
cd "$(dirname "$0")"
while pgrep -f "train_lora[.]py.*piper_s1" >/dev/null; do sleep 30; done
for p in $(ps aux | grep "[r]un_panda_record.sh" | awk '{print $2}'); do kill $p; done
for p in $(ps aux | grep "[s]witch_experiment.py --policy outputs/v6c3" | awk '{print $2}'); do kill $p; done
echo "=== $(date +%m/%d\ %H:%M) Panda 기록 평가 멈춤 → run_panda_record2.sh"
./run_panda_record2.sh >> outputs/v6/panda_record.log 2>&1
