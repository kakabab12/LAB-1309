#!/usr/bin/env bash
# retreat ablation: 무엇이 효과의 핵심인가 (내려놓기만 / 위치만 / 회전만 / 전부)
# 전부(retreat)와 flush는 9/14 스윕 결과를 그대로 비교에 씀
set -u
cd "$(dirname "$0")"
J=outputs/abl_jobs.txt; : > $J
for s in ret_rot ret_pos release; do
  for p in 8:7 4:7 1:5 2:5; do
    for k in grasp:3 grasp:20; do
      echo "--task-a ${p%:*} --task-b ${p#*:} --switch-at $k --strategy $s" >> $J
    done
  done
done
cat $J | xargs -P 3 -I{} sh -c '.venv/bin/python switch_experiment.py {} --episodes 10 --out outputs/switch 2>&1 | grep -E "^SUMMARY|Traceback|Error"'
echo DONE
