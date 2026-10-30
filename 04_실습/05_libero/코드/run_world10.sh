#!/usr/bin/env bash
# w10 (10/30 새벽 계획): 물체 N개 끼어들기를 배우게 — 판 비우기를 넣은 시범(world_expert.clear_dest)으로 끼어들기 DAgger 깊이 2·3.
#   사용자 "연구장면도 2개만 옮기는 게 아니라 n개 다 옮길 수 있게". w8 + 다듬기 일 2개에서 VLA 가 A 를 B 의 판 위에 내려놓은 것이 43번 중 22번,
#   예전 시범은 그 위에 B 를 그대로 놓았다 → 이제 판 위 다른 물체를 먼저 빈 자리로 치운다 (VLA 없는 시험 3/3).
#   ⓪ w9 와 w8 중 일 1개 다듬기(3cm) 가 나은 것을 고른다 (C). w9 + 새 다듬기 다 켬(w9ra) 남은 장면을 마저 잰다.
#   ① 끼어들기 DAgger 깊이 2 (world_dagger_nested, 세 설정 × 120, 14000~) ‖ 깊이 3 (world_dagger_deep, 세 설정 × 40, 15000~)
#   ② 사진 특징·위치 입력 (vis_cache, wfeat2sc → wfeat3sc)  ③ w10 학습 (C 에서 15000, 지금까지 데이터 + 새 DAgger)
#   ④ w10 + 다듬기 다 켬: 일 1개 50 (세 설정), 일 2개·3개 끼어들기 20 (세 설정) / w10 + 예전 다듬기·혼자 일 1개 50 (비교)
#   GPU 는 한꺼번에 4개까지 (DAgger 처럼 지시문이 여러 개면 하나가 2.9GB). 끄기는 PID 나 systemctl --user stop 으로만.
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
RA="grasp,place,fmask,settle,wxy,gplan,objc,pfine"
GP=()
gp(){ ( "$@" ) & GP+=($!); sleep 60; }            # GPU 하나 띄우기 (메모리가 자리 잡게 1분)
gpwait(){ wait "${GP[@]}"; GP=(); }
okcount(){ $PY -c "
import json,sys
n=0
for c in ('default','random','many'):
    try: n+=sum(e['all_ok'] for e in json.load(open(f'outputs/world_{sys.argv[1]}_{c}/k1n.json')))
    except Exception: pass
print(n)" $1; }
wf2c(){ for s in 0 1 2; do ( PIPER_WORLD=piper_sim/worlds/$2.json nice -n 10 $PY piper_sim/world_feat_cache.py --data $1 --shard $s --nshard 3 --stride 6 --version 2 --sure --tag c 2>&1 | grep -E "^끝|Traceback|Error" ) & done; wait; }

C8=outputs/piper_w8_model/merged; C9=outputs/piper_w9_model/merged
a9=$(okcount w9r); a8=$(okcount w8rm)
if [ -f $C9/worldfeat_cammask ] && [ "$a9" -ge "$a8" ]; then C=$C9; CN=w9; else C=$C8; CN=w8; fi
log "[⓪] 고른 모델 $CN (일 1개 다듬기 3cm: w9r $a9 · w8rm $a8 / 150)"

log "[⓪ w9 + 다듬기 다 켬 남은 장면] ‖ [① 깊이 3 DAgger 기본]"
for c in default random many; do gp env PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_eval.py --policy $C9 --seeds 1000-1049 --k 1 --nested $EVQ --refine $RA --out outputs/world_w9ra_$c/k1n.json > outputs/piper/w10_w9ra_$c.log 2>&1; done
gp env PIPER_WORLD=piper_sim/worlds/default.json $PY piper_sim/world_dagger_deep.py --policy $C --depth 3 --seeds 15000-15039 --out data/world_daggerK3_default $EVQ > outputs/piper/world_daggerK3_default.log 2>&1
gpwait
grep -h SUMMARY outputs/piper/w10_w9ra_*.log
log "[① 끼어들기 DAgger 깊이 2] $CN, 세 설정 × 120"
i=0
for c in default random many; do
  a=$((14000 + i * 200)); b=$((a + 119)); i=$((i + 1))
  gp env PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_dagger_nested.py --policy $C --seeds $a-$b --out data/world_daggerN2_$c $EVQ > outputs/piper/world_daggerN2_$c.log 2>&1
done
gpwait                                             # 끼어들기 DAgger 는 지시문이 여러 개라 하나가 2.9GB — 4개를 같이 띄우면 11.3GB 를 넘는다
log "[① 깊이 3 흔든·물체 많음]"
gp env PIPER_WORLD=piper_sim/worlds/random.json $PY piper_sim/world_dagger_deep.py --policy $C --depth 3 --seeds 15100-15139 --out data/world_daggerK3_random $EVQ > outputs/piper/world_daggerK3_random.log 2>&1
gp env PIPER_WORLD=piper_sim/worlds/many.json $PY piper_sim/world_dagger_deep.py --policy $C --depth 3 --seeds 15200-15239 --out data/world_daggerK3_many $EVQ > outputs/piper/world_daggerK3_many.log 2>&1
gpwait
grep -h STATS outputs/piper/world_daggerN2_*.log outputs/piper/world_daggerK3_*.log

NEW="data/world_daggerN2_default data/world_daggerN2_random data/world_daggerN2_many data/world_daggerK3_default data/world_daggerK3_random data/world_daggerK3_many"
log "[② 새 DAgger 사진 특징 + 위치 입력]"
VP=()
for s in 0 1; do
  ( $PY vis_cache.py build --policy outputs/piper_np3_model/merged --data $NEW --out cache/vis --batch 8 --stride 6 --shard $s --nshard 2 2>&1 | grep -E "끝|Error|Traceback" ) &
  VP+=($!)
done
for c in default random many; do wf2c data/world_daggerN2_$c $c; wf2c data/world_daggerK3_$c $c; done
wait "${VP[@]}"
D45="data/world_dagger4_default data/world_dagger4_random data/world_dagger4_many data/world_dagger5_default data/world_dagger5_random data/world_dagger5_many"
DN="data/world_daggerN_default data/world_daggerN_random data/world_daggerN_many"
D6="data/world_dagger6_default data/world_dagger6_random data/world_dagger6_many"
DATA="data/world_default_t data/world_random data/world_many data/world_dagger_d1 data/world_dagger_r1 data/world_dagger2_default data/world_dagger2_random data/world_dagger2_many data/world_dagger3_default data/world_dagger3_random data/world_dagger3_many $D45 $DN $D6 $NEW"
PIPER_WORLD=piper_sim/worlds/default.json $PY piper_sim/world_feat_cache.py --data $NEW --version 3 --from-v2 --sure --tag c | tail -1
for d in $DATA; do
  a=$(ls $d/episodes/*.wfeat3sc.npy | wc -l); b=$(ls $d/episodes/*.npz | wc -l)
  [ "$a" -eq "$b" ] || { log "$d wfeat3sc $a / $b 모자람 — 멈춤"; exit 1; }
done

log "[③ w10 학습] $CN 에서 15000스텝, + 끼어들기 DAgger 깊이 2·3 (판 비우기) (MemAvailable $(grep MemAvailable /proc/meminfo | awk '{print int($2/1e6)}')GB)"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --log-every 500 --abs-pos --objfeat --feat wfeat3sc --renorm-new-feat --renorm-action --vis-cache cache/vis --frame-stride 6 --sufstate --fp32-master --vlm-lora 16 --vlm-lora-lr 2e-4 --proj-lr 3e-4 --new-lr 3e-4"
$PY train_lora.py --policy $C --data $DATA $TR --batch-size 8 --grad-accum 2 --steps 15000 --lr 3e-5 --eval-every 5000 --save-every 5000 \
    --out outputs/piper_w10_model 2>&1 | grep --line-buffered -E '에피소드 학습|새로 쓰는|"step": [0-9]*000, "loss"|val_loss|모델 저장|Traceback|Error|Killed'
CW=outputs/piper_w10_model/merged
[ -f $CW/worldfeat_sure ] && [ -f $CW/worldfeat_v3 ] && [ -f $CW/worldfeat_cammask ] || { log "w10 모델 표시 없음 — 멈춤"; exit 1; }

ev(){ gp env PIPER_WORLD=piper_sim/worlds/$1.json $PY piper_sim/world_eval.py --policy $CW --seeds $2 --k $4 --nested $EVQ $3 --out outputs/world_w10$5_$1/k$4n.json > outputs/piper/w10_$5_$1_k$4.log 2>&1; }
log "[④ w10 + 다듬기 다 켬] 일 1개 50장면 ‖ 예전 다듬기 기본"
for c in default random many; do ev $c 1000-1049 "--refine $RA" 1 ra; done
ev default 1000-1049 "--refine grasp,place" 1 r
gpwait
log "[④ w10] 예전 다듬기 흔든·많음 ‖ 혼자 기본·흔든"
ev random 1000-1049 "--refine grasp,place" 1 r; ev many 1000-1049 "--refine grasp,place" 1 r
ev default 1000-1049 "" 1 ""; ev random 1000-1049 "" 1 ""
gpwait
log "[④ w10 + 다듬기 다 켬] 일 2개 끼어들기 20장면"          # 지시문이 여러 개 → 하나가 2.9GB, 3개까지만
for c in default random many; do ev $c 1000-1019 "--refine $RA" 2 ra; done
gpwait
log "[④ w10 + 다듬기 다 켬] 일 3개 끼어들기 20장면"
for c in default random many; do ev $c 1000-1019 "--refine $RA" 3 ra; done
gpwait
log "[④ w10] 혼자 물체 많음"
ev many 1000-1049 "" 1 ""
gpwait
grep -h SUMMARY outputs/piper/w10_*.log
for c in default random many; do PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_table.py outputs/world_w9r_$c outputs/world_w9ra_$c outputs/world_w10_$c outputs/world_w10r_$c outputs/world_w10ra_$c; done
log "끝"
