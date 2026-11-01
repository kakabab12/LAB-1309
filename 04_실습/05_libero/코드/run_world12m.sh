#!/usr/bin/env bash
# w12m (11/2): w11m 이 실제로 가 있는 상태에서 DAgger 한 번 더 + '실수하는 순간 이어받기' (--stop-release).
#   11/2 w10m + RC 기본 쌓기 실패 따라가기 (1006·1021·1028): VLA 가 지시한 물체를 쥐고 쌓을 목표 10~20cm 앞에서 멈춰 내려가 손을 편다
#   (위치 입력은 목표까지 9.6cm 로 맞음, 다듬기는 손대지 않음). 무작위 T 에서만 이어받으면 그 순간의 바로잡기 시범이 드물다
#   → VLA 가 지시한 물체를 쥔 채 놓을 곳 4cm 밖에서 펴려는 스텝에 멈추고 시범 프로그램이 그 자리에서 옮겨 놓는다 (HG-DAgger 처럼).
#   VLA 구간은 평가와 같이 다듬기(RC)를 켠 채 (--refine): VLA 혼자 돌리면 쥐기를 못 해 옮기는 단계의 실수가 데이터에 안 담긴다.
#   ① DAgger (w11m, GPU 일꾼 셋): 일 1개 --fix-wrong --stop-release 4 (쌓기 50%, VLA 60~450스텝) 세 설정 × 150 (31000~),
#      깊이 2 --ret-p 0.6 --stop-release 4 세 설정 × 60 (32000~), 깊이 3 같은 옵션 세 설정 × 20 (33000~)
#   ② 사진 특징·위치 입력  ③ w12m 학습 (w11m 에서 15000스텝, 지시문 바꿔 쓰기 50%)  ④ 평가 (w11m ④ 와 같은 장면·RC 다듬기)
#   w11m 학습·평가(piper-w11mc, 로그 W11LOG)가 끝난 뒤 시작. 끄기는 systemctl --user stop piper-w12m 으로만.
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"
CD=outputs/piper_w11m_model/merged
RC=$(cat outputs/piper/ra_choice.txt)                               # 평가와 같은 다듬기를 DAgger 의 VLA 구간에도 (11/2 시험: VLA 혼자면 쌓기 3장면 모두 300스텝 안에 못 쥠)
LOCK=outputs/piper/gpu_launch.lock
CL=outputs/piper/claim_w12m
gpu_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
gpu_n(){ nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -c python; }
mem_av(){ awk '/MemAvailable/{print int($2/1e6)}' /proc/meminfo; }
held=0
trap '[ $held -eq 1 ] && rmdir $LOCK 2>/dev/null; exit 0' EXIT TERM INT

run_item(){ IFS=: read -r kind c a b d e f <<< "$1"
  case $kind in
    ev) local opt=""; [ "$d" != "-" ] && opt="--refine $d"
        env PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_eval.py --policy $f --seeds $b --k $a --nested $EVQ $opt \
            --out outputs/world_${e}_$c/k${a}n.json > outputs/piper/w12m_${e}_${c}_k$a.log 2>&1 ;;
    x1) env PIPER_WORLD=piper_sim/worlds/$c.json PIPER_STACK_P=0.5 $PY piper_sim/world_dagger.py --policy $CD --seeds $a --out data/world_daggerX1_$c \
            --fix-wrong --stop-release 4 --t 60 450 --refine $RC $EVQ > outputs/piper/w12m_x1_$c.log 2>&1 ;;
    x2) env PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_dagger_nested.py --policy $CD --seeds $a --out data/world_daggerX2_$c \
            --ret-p 0.6 --stop-release 4 --refine $RC $EVQ > outputs/piper/w12m_x2_$c.log 2>&1 ;;
    x3) env PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_dagger_deep.py --policy $CD --depth 3 --seeds $a --out data/world_daggerX3_$c \
            --ret-p 0.6 --stop-release 4 --refine $RC $EVQ > outputs/piper/w12m_x3_$c.log 2>&1 ;;
  esac; }
gworker(){ for it in "$@"; do mkdir $CL/${it//[:\/,]/_} 2>/dev/null || continue
    until mkdir $LOCK 2>/dev/null; do sleep 15; done
    while [ "$(gpu_free)" -lt 3400 ] || [ "$(gpu_n)" -ge 3 ] || [ "$(mem_av)" -lt 6 ]; do sleep 30; done
    log "띄움 $it (GPU 남은 $(gpu_free)MiB, 프로세스 $(gpu_n), 메모리 $(mem_av)GB)"
    run_item "$it" & local P=$!
    sleep 120; rmdir $LOCK
    wait $P; log "끝 $it"
  done; }
mkdir -p $CL

W11LOG=${W11LOG:-outputs/piper/world11mc.log}                        # 11/2 1시: w11m 은 run_world11m_c.sh (쥔 물체 따라가기 입력) 로 학습·평가
until tail -1 $W11LOG 2>/dev/null | grep -qE "^=== [0-9/]+ [0-9:]+ 끝$|멈춤"; do sleep 120; done   # 마지막 줄이 '=== 날짜 시각 끝' (ⓐ 끝·ⓑ 끝 말고)
tail -1 $W11LOG | grep -q "멈춤" && { log "w11m 파이프라인 멈춤 — 그만"; exit 1; }
[ -f $CD/worldfeat_v3 ] || { log "w11m 모델 없음 — 그만"; exit 1; }

log "[① w12m DAgger] 시작 모델 $CD (GPU 일꾼 셋, 실수하는 순간 이어받기 4cm)"
D1="x1:default:31000-31149 x1:random:31200-31349 x1:many:31400-31549
    x2:default:32000-32059 x2:random:32200-32259 x2:many:32400-32459
    x3:default:33000-33019 x3:random:33100-33119 x3:many:33200-33219"
W=()
for i in 1 2 3; do gworker $D1 & W+=($!); sleep 5; done
wait "${W[@]}"
grep -h STATS outputs/piper/w12m_x*.log

NEW="data/world_daggerX1_default data/world_daggerX1_random data/world_daggerX1_many data/world_daggerX2_default data/world_daggerX2_random data/world_daggerX2_many
     data/world_daggerX3_default data/world_daggerX3_random data/world_daggerX3_many"
for d in $NEW; do [ -d $d/episodes ] || { log "$d 없음 — 멈춤"; exit 1; }; done
log "[② 새 데이터 사진 특징 + 위치 입력] $(for d in $NEW; do echo -n "$(basename $d) $(ls $d/episodes/*.npz | wc -l) "; done)"
VP=()
for s in 0 1; do
  ( $PY vis_cache.py build --policy outputs/piper_np3_model/merged --data $NEW --out cache/vis --batch 8 --stride 6 --shard $s --nshard 2 2>&1 | grep -E "끝|Error|Traceback" ) &
  VP+=($!); sleep 20
done
wf2c(){ local FP=(); for s in 0 1 2; do ( PIPER_WORLD=piper_sim/worlds/$2.json nice -n 10 $PY piper_sim/world_feat_cache.py --data $1 --shard $s --nshard 3 --stride 6 --version 2 --sure --tag c 2>&1 | grep -E "^끝|Traceback|Error" ) & FP+=($!); sleep 20; done; wait "${FP[@]}"; }
for c in default random many; do wf2c data/world_daggerX1_$c $c; wf2c data/world_daggerX2_$c $c; wf2c data/world_daggerX3_$c $c; done
wait "${VP[@]}"
PIPER_WORLD=piper_sim/worlds/default.json $PY piper_sim/world_feat_cache.py --data $NEW --version 3 --from-v2 --sure --tag c | tail -1
PIPER_WORLD=piper_sim/worlds/default.json $PY piper_sim/world_feat_cache.py --data $NEW --version 3 --from-v2 --sure --tag c --held --stride 6 | tail -1   # 11/2 쥔 물체 따라가기 (wfeat3sch)
FEAT=$([ -f $CD/worldfeat_held ] && echo wfeat3sch || echo wfeat3sc)          # w11m 이 쥔 물체 따라가기로 학습됐으면 그대로 이어서
D45="data/world_dagger4_default data/world_dagger4_random data/world_dagger4_many data/world_dagger5_default data/world_dagger5_random data/world_dagger5_many"
DN="data/world_daggerN_default data/world_daggerN_random data/world_daggerN_many"
D6="data/world_dagger6_default data/world_dagger6_random data/world_dagger6_many"
N10="data/world_motion_default data/world_motion_random data/world_motion_many data/world_daggerM2_default data/world_daggerM2_random data/world_daggerM2_many
     data/world_daggerM3_default data/world_daggerM3_random data/world_daggerM3_many data/world_daggerK3_default"
N11="data/world_motion2_default data/world_motion2_random data/world_motion2_many data/world_daggerW1_default data/world_daggerW1_random data/world_daggerW1_many
     data/world_daggerW2_default data/world_daggerW2_random data/world_daggerW2_many data/world_daggerW3_default data/world_daggerW3_random data/world_daggerW3_many"
DATA="data/world_default_t data/world_random data/world_many data/world_dagger_d1 data/world_dagger_r1 data/world_dagger2_default data/world_dagger2_random data/world_dagger2_many data/world_dagger3_default data/world_dagger3_random data/world_dagger3_many $D45 $DN $D6 $N10 $N11 $NEW"
for d in $DATA; do
  a=$(ls $d/episodes/*.$FEAT.npy | wc -l); b=$(ls $d/episodes/*.npz | wc -l)
  [ "$a" -eq "$b" ] || { log "$d $FEAT $a / $b 모자람 — 멈춤"; exit 1; }
done

until mkdir $LOCK 2>/dev/null; do sleep 15; done; held=1
until [ "$(gpu_n)" -eq 0 ]; do sleep 30; done
log "[③ w12m 학습] $CD 에서 15000스텝, 위치 입력 $FEAT, + 새 DAgger (실수 순간 이어받기), 지시문 바꿔 쓰기 50% (MemAvailable $(mem_av)GB) — 자물쇠 쥠"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --log-every 500 --abs-pos --objfeat --feat $FEAT --renorm-new-feat --renorm-action --vis-cache cache/vis --frame-stride 6 --sufstate --fp32-master --vlm-lora 16 --vlm-lora-lr 2e-4 --proj-lr 3e-4 --new-lr 3e-4"
PIPER_WORLD=piper_sim/worlds/default.json $PY train_lora.py --policy $CD --data $DATA $TR --instr-aug 0.5 --batch-size 8 --grad-accum 2 --steps 15000 --lr 3e-5 --eval-every 5000 --save-every 5000 \
    --out outputs/piper_w12m_model 2>&1 | grep --line-buffered -E '에피소드 학습|새로 쓰는|"step": [0-9]*000, "loss"|val_loss|모델 저장|Traceback|Error|Killed'
rmdir $LOCK; held=0
CW=outputs/piper_w12m_model/merged
[ -f $CW/worldfeat_sure ] && [ -f $CW/worldfeat_v3 ] && [ -f $CW/worldfeat_cammask ] || { log "w12m 모델 표시 없음 — 멈춤"; exit 1; }

log "[④ w12m 새 기준 평가] 다듬기 $RC (GPU 일꾼 셋)"
E4="ev:default:1:1000-1049:$RC:w12mraM:$CW ev:random:1:1000-1049:$RC:w12mraM:$CW ev:many:1:1000-1049:$RC:w12mraM:$CW
    ev:default:2:1000-1019:$RC:w12mraM:$CW ev:default:1:1000-1049:-:w12mM:$CW ev:random:1:1000-1049:-:w12mM:$CW ev:many:1:1000-1049:-:w12mM:$CW
    ev:random:2:1000-1019:$RC:w12mraM:$CW ev:many:2:1000-1019:$RC:w12mraM:$CW
    ev:default:3:1000-1019:$RC:w12mraM:$CW ev:random:3:1000-1019:$RC:w12mraM:$CW ev:many:3:1000-1019:$RC:w12mraM:$CW"
W=()
for i in 1 2 3; do gworker $E4 & W+=($!); sleep 5; done
wait "${W[@]}"
grep -h SUMMARY outputs/piper/w12m_w12m*_k*.log
for c in default random many; do PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_table.py outputs/world_w11mM_$c outputs/world_w11mraM_$c outputs/world_w12mM_$c outputs/world_w12mraM_$c; done
log "끝"
