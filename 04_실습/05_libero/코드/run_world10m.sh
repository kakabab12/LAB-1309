#!/usr/bin/env bash
# w10m (10/30 낮): 판 N개 + 동작 변수 — 사용자 '판도 n개로', '동작을 여러 변수로 (더 좋은 게 있으면 추가)', '하드코딩 아닌 설정 파일로'.
#   판 개수 = 설정 n_pads 범위 (world.pick_n_pads, 예전 평가는 판 2개 고정). 동작 변수 = 설정 motion (world.sample_motion·vary_physics):
#   끼어드는 때 (다가가는 중 3~10cm·쥔 직후 1~10스텝·옮기는 중 20~80스텝·놓기 직전 3~8cm), 일 종류 (쌓기 25%), 지시문 틀 (판 5·쌓기 4),
#   지연 6~12스텝, 속도 0.8~1.2배, 제어 잡음, 물체 마찰·무게, 손가락 마찰. 결과 이름 끝 M = 이 기준 (예전 숫자와 섞지 않음).
#   ⓪ w9 를 새 기준으로: 혼자·다듬기 다 켬 일 1개 50장면 세 설정, 다듬기 일 2·3개 기본 20장면
#   ① 새 데이터 (⓪ 과 같이): 시범 프로그램 세 설정 × 100 (CPU 2줄), 끼어들기 DAgger 깊이 2 세 설정 × 60·깊이 3 × 20 (w9)
#   ② 사진 특징·위치 입력  ③ w10m 학습 (w9 에서 15000, 지금까지 데이터 + 새 데이터, 지시문 틀 바꿔 쓰기 50%)
#   ④ w10m 평가 (⓪ 과 같은 장면 + 일 2·3개 세 설정)
#   GPU 는 자물쇠(outputs/piper/gpu_launch.lock) + 남은 메모리 3.2GB·GPU 프로세스 4개 미만일 때만 띄움.
#   끄기는 systemctl --user stop piper-w10m 으로만 (안의 프로세스가 다 같이 꺼진다).
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"      # 지연은 장면마다 설정 motion 에서 (9 는 motion 을 껐을 때만)
RA="grasp,place,fmask,settle,wxy,gplan,objc,pfine"
C9=outputs/piper_w9_model/merged
LOCK=outputs/piper/gpu_launch.lock
CL=outputs/piper/claim_w10m
gpu_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
gpu_n(){ nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .; }
mem_av(){ awk '/MemAvailable/{print int($2/1e6)}' /proc/meminfo; }

# 항목 하나 실행 (이름으로). ev:<설정>:<k>:<장면>:<다듬기|->:<이름>:<정책>, dn:<설정>:<장면>, dk:<설정>:<장면>, co:<설정>:<장면>
run_item(){ IFS=: read -r kind c a b d e f <<< "$1"
  case $kind in
    ev) local opt=""; [ "$d" != "-" ] && opt="--refine $d"
        env PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_eval.py --policy $f --seeds $b --k $a --nested $EVQ $opt \
            --out outputs/world_${e}_$c/k${a}n.json > outputs/piper/w10m_${e}_${c}_k$a.log 2>&1 ;;
    dn) env PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_dagger_nested.py --policy $C9 --seeds $a --out data/world_daggerM2_$c $EVQ \
            > outputs/piper/w10m_dn_$c.log 2>&1 ;;
    dk) env PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_dagger_deep.py --policy $C9 --depth 3 --seeds $a --out data/world_daggerM3_$c $EVQ \
            > outputs/piper/w10m_dk_$c.log 2>&1 ;;
    co) env PIPER_WORLD=piper_sim/worlds/$c.json nice -n 5 $PY piper_sim/world_collect.py --seeds $a --k 1 4 --out data/world_motion_$c \
            > outputs/piper/w10m_co_${c}_${a}.log 2>&1 ;;
  esac; }
# GPU 일꾼: 항목을 다른 일꾼과 나눠 가진다 (mkdir 로 먼저 잡은 쪽). 띄우기는 자물쇠 + 메모리·개수 확인, 띄운 뒤 2분 기다려 자물쇠를 놓는다
gworker(){ for it in "$@"; do mkdir $CL/${it//[:\/,]/_} 2>/dev/null || continue
    until mkdir $LOCK 2>/dev/null; do sleep 15; done
    while [ "$(gpu_free)" -lt 3200 ] || [ "$(gpu_n)" -ge 4 ] || [ "$(mem_av)" -lt 6 ]; do sleep 30; done
    log "띄움 $it (GPU 남은 $(gpu_free)MiB, 프로세스 $(gpu_n), 메모리 $(mem_av)GB)"
    run_item "$it" & local P=$!
    sleep 120; rmdir $LOCK
    wait $P; log "끝 $it"
  done; }
cworker(){ for it in "$@"; do mkdir $CL/${it//[:\/,]/_} 2>/dev/null || continue
    while [ "$(mem_av)" -lt 5 ]; do sleep 60; done
    log "띄움 $it (CPU, 메모리 $(mem_av)GB)"; run_item "$it"; log "끝 $it"
  done; }
rm -rf $CL; mkdir -p $CL

E0="ev:default:1:1000-1049:$RA:w9raM:$C9 ev:random:1:1000-1049:$RA:w9raM:$C9 ev:many:1:1000-1049:$RA:w9raM:$C9
    ev:default:1:1000-1049:-:w9M:$C9 ev:random:1:1000-1049:-:w9M:$C9 ev:many:1:1000-1049:-:w9M:$C9
    ev:default:2:1000-1019:$RA:w9raM:$C9 ev:default:3:1000-1019:$RA:w9raM:$C9"
D1="dn:default:16000-16059 dn:random:16200-16259 dn:many:16400-16459 dk:default:17000-17019 dk:random:17100-17119 dk:many:17200-17219"
C1="co:default:20000-20049 co:random:20200-20249 co:many:20400-20449 co:default:20050-20099 co:random:20250-20299 co:many:20450-20499"
log "[⓪ w9 새 기준] ‖ [① 새 데이터: DAgger 깊이 2·3 (GPU) · 시범 (CPU)]  (GPU 일꾼 넷: 평가 둘·DAgger 둘, CPU 일꾼 둘)"
W=()
gworker $E0 & W+=($!); sleep 5
gworker $E0 & W+=($!); sleep 5
gworker $D1 & W+=($!); sleep 5
gworker $D1 & W+=($!); sleep 5
cworker $C1 & W+=($!); sleep 5
cworker $C1 & W+=($!)
wait "${W[@]}"
grep -h SUMMARY outputs/piper/w10m_w9*_k*.log
grep -h STATS outputs/piper/w10m_dn_*.log outputs/piper/w10m_dk_*.log outputs/piper/w10m_co_*.log

NEW="data/world_motion_default data/world_motion_random data/world_motion_many data/world_daggerM2_default data/world_daggerM2_random data/world_daggerM2_many
     data/world_daggerM3_default data/world_daggerM3_random data/world_daggerM3_many data/world_daggerK3_default"
for d in $NEW; do [ -d $d/episodes ] || { log "$d 없음 — 멈춤"; exit 1; }; done
log "[② 새 데이터 사진 특징 + 위치 입력] $(for d in $NEW; do echo -n "$(basename $d) $(ls $d/episodes/*.npz | wc -l) "; done)"
VP=()
for s in 0 1; do
  ( $PY vis_cache.py build --policy outputs/piper_np3_model/merged --data $NEW --out cache/vis --batch 8 --stride 6 --shard $s --nshard 2 2>&1 | grep -E "끝|Error|Traceback" ) &
  VP+=($!)
done
wf2c(){ for s in 0 1 2; do ( PIPER_WORLD=piper_sim/worlds/$2.json nice -n 10 $PY piper_sim/world_feat_cache.py --data $1 --shard $s --nshard 3 --stride 6 --version 2 --sure --tag c 2>&1 | grep -E "^끝|Traceback|Error" ) & done; wait; }
for c in default random many; do wf2c data/world_motion_$c $c; wf2c data/world_daggerM2_$c $c; wf2c data/world_daggerM3_$c $c; done
wf2c data/world_daggerK3_default default
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

log "[③ w10m 학습] w9 에서 15000스텝, + 새 데이터 (동작 변수), 지시문 틀 바꿔 쓰기 50% (MemAvailable $(mem_av)GB)"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --log-every 500 --abs-pos --objfeat --feat wfeat3sc --renorm-new-feat --renorm-action --vis-cache cache/vis --frame-stride 6 --sufstate --fp32-master --vlm-lora 16 --vlm-lora-lr 2e-4 --proj-lr 3e-4 --new-lr 3e-4"
PIPER_WORLD=piper_sim/worlds/default.json $PY train_lora.py --policy $C9 --data $DATA $TR --instr-aug 0.5 --batch-size 8 --grad-accum 2 --steps 15000 --lr 3e-5 --eval-every 5000 --save-every 5000 \
    --out outputs/piper_w10m_model 2>&1 | grep --line-buffered -E '에피소드 학습|새로 쓰는|"step": [0-9]*000, "loss"|val_loss|모델 저장|Traceback|Error|Killed'
CW=outputs/piper_w10m_model/merged
[ -f $CW/worldfeat_sure ] && [ -f $CW/worldfeat_v3 ] && [ -f $CW/worldfeat_cammask ] || { log "w10m 모델 표시 없음 — 멈춤"; exit 1; }

log "[④ w10m 새 기준 평가] (GPU 일꾼 넷)"
E4="ev:default:1:1000-1049:$RA:w10mraM:$CW ev:random:1:1000-1049:$RA:w10mraM:$CW ev:many:1:1000-1049:$RA:w10mraM:$CW
    ev:default:1:1000-1049:-:w10mM:$CW ev:random:1:1000-1049:-:w10mM:$CW ev:many:1:1000-1049:-:w10mM:$CW
    ev:default:2:1000-1019:$RA:w10mraM:$CW ev:random:2:1000-1019:$RA:w10mraM:$CW ev:many:2:1000-1019:$RA:w10mraM:$CW
    ev:default:3:1000-1019:$RA:w10mraM:$CW ev:random:3:1000-1019:$RA:w10mraM:$CW ev:many:3:1000-1019:$RA:w10mraM:$CW"
W=()
for i in 1 2 3 4; do gworker $E4 & W+=($!); sleep 5; done
wait "${W[@]}"
grep -h SUMMARY outputs/piper/w10m_w10m*_k*.log
for c in default random many; do PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_table.py outputs/world_w9M_$c outputs/world_w9raM_$c outputs/world_w10mM_$c outputs/world_w10mraM_$c; done
log "끝"
