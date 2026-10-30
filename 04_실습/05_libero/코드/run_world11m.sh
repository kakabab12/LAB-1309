#!/usr/bin/env bash
# w11m (10/31 새벽): 새 기준(판 N개 + 동작 변수)에서 w10m 이 실제로 가 있는 상태로 DAgger 를 한 번 더.
#   w9 새 기준 실패: 쌓기·낯선 지시문에서 지시한 물체를 안 쥠 (대부분 아무것도 안 쥠), 일 2개에서 B 를 마친 뒤 A 로 못 돌아옴 (5/5).
#   ① DAgger (GPU, w10m ④ 평가와 같은 자물쇠로 끼어 돈다):
#      일 1개 world_dagger --fix-wrong (다른 물체를 쥐었으면 내려놓고 지시한 물체로) 세 설정 × 100 (21000~)
#      깊이 2 world_dagger_nested --ret-p 0.6 (B 를 마친 뒤 VLA 가 돌아오기를 먼저 해 보고 이어받음) 세 설정 × 60 (22000~)
#      깊이 3 world_dagger_deep --ret-p 0.6 세 설정 × 20 (23000~)
#      + 시범 프로그램 세 설정 × 80, 쌓기 50% (piper-co11 → data/world_motion2_*, 여기서는 끝나기만 기다림)
#   ② 사진 특징·위치 입력  ③ w11m 학습 (w10m 에서 15000, 지시문 바꿔 쓰기 50%, 쌓기 몫 15% — outputs/train_override.json)
#   ④ w11m 평가 (w10m ④ 와 같은 장면: 일 1개 다듬기 다 켬·혼자 세 설정 50, 일 2·3개 다 켬 세 설정 20)
#   GPU 는 자물쇠(outputs/piper/gpu_launch.lock) + 남은 메모리 3.2GB·GPU 프로세스 4개 미만일 때만 띄움.
#   끄기는 systemctl --user stop piper-w11m 으로만 (안의 프로세스가 다 같이 꺼진다).
cd "$(dirname "$0")"
export PIPER_NOPED=1 MUJOCO_GL=egl
PY=.venv/bin/python
log(){ echo "=== $(date +%m/%d\ %H:%M) $*"; }
EVQ="--latency-steps 9 --ttrtc --n-action-steps 1 --cuda-graph"      # 지연은 장면마다 설정 motion 에서
RA="${RA:-grasp,place,fmask,settle,wxy,gplan,objc,pfine}"           # amb 비교를 보고 띄울 때 정함
CD=${CD:-outputs/piper_w10m_model/merged}                           # DAgger·학습 시작 모델
LOCK=outputs/piper/gpu_launch.lock
CL=outputs/piper/claim_w11m
gpu_free(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
gpu_n(){ nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .; }
mem_av(){ awk '/MemAvailable/{print int($2/1e6)}' /proc/meminfo; }

# 항목 하나 실행. ev:<설정>:<k>:<장면>:<다듬기|->:<이름>:<정책>, d1/dn/dk:<설정>:<장면>
run_item(){ IFS=: read -r kind c a b d e f <<< "$1"
  case $kind in
    ev) local opt=""; [ "$d" != "-" ] && opt="--refine $d"
        env PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_eval.py --policy $f --seeds $b --k $a --nested $EVQ $opt \
            --out outputs/world_${e}_$c/k${a}n.json > outputs/piper/w11m_${e}_${c}_k$a.log 2>&1 ;;
    d1) env PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_dagger.py --policy $CD --seeds $a --out data/world_daggerW1_$c --fix-wrong $EVQ \
            > outputs/piper/w11m_d1_$c.log 2>&1 ;;
    dn) env PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_dagger_nested.py --policy $CD --seeds $a --out data/world_daggerW2_$c --ret-p 0.6 $EVQ \
            > outputs/piper/w11m_dn_$c.log 2>&1 ;;
    dk) env PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_dagger_deep.py --policy $CD --depth 3 --seeds $a --out data/world_daggerW3_$c --ret-p 0.6 $EVQ \
            > outputs/piper/w11m_dk_$c.log 2>&1 ;;
  esac; }
gworker(){ for it in "$@"; do mkdir $CL/${it//[:\/,]/_} 2>/dev/null || continue
    until mkdir $LOCK 2>/dev/null; do sleep 15; done
    while [ "$(gpu_free)" -lt 3200 ] || [ "$(gpu_n)" -ge 4 ] || [ "$(mem_av)" -lt 6 ]; do sleep 30; done
    log "띄움 $it (GPU 남은 $(gpu_free)MiB, 프로세스 $(gpu_n), 메모리 $(mem_av)GB)"
    run_item "$it" & local P=$!
    sleep 120; rmdir $LOCK
    wait $P; log "끝 $it"
  done; }
mkdir -p $CL                                         # 다시 띄우면 이미 잡은 항목은 건너뜀 (처음부터면 claim 폴더를 지울 것)

D1="d1:default:21000-21099 d1:random:21200-21299 d1:many:21400-21499
    dn:default:22000-22059 dn:random:22200-22259 dn:many:22400-22459
    dk:default:23000-23019 dk:random:23100-23119 dk:many:23200-23219"
until grep -qE "④|멈춤" outputs/piper/world10m.log; do sleep 60; done   # w10m 학습이 끝나 평가 첫 셋이 자리 잡은 뒤
grep -q "멈춤" outputs/piper/world10m.log && { log "w10m 파이프라인 멈춤 — 그만"; exit 1; }
sleep 900
log "[① w11m DAgger] 시작 모델 $CD (GPU 일꾼 셋 — w10m 평가와 자물쇠를 나눠 씀)"
W=()
for i in 1 2 3; do gworker $D1 & W+=($!); sleep 5; done
wait "${W[@]}"
grep -h STATS outputs/piper/w11m_d1_*.log outputs/piper/w11m_dn_*.log outputs/piper/w11m_dk_*.log
until grep -qE "^=== .* 끝$" outputs/piper/co11.log 2>/dev/null; do sleep 120; done
grep -h STATS outputs/piper/co11_*.log

NEW="data/world_motion2_default data/world_motion2_random data/world_motion2_many data/world_daggerW1_default data/world_daggerW1_random data/world_daggerW1_many
     data/world_daggerW2_default data/world_daggerW2_random data/world_daggerW2_many data/world_daggerW3_default data/world_daggerW3_random data/world_daggerW3_many"
for d in $NEW; do [ -d $d/episodes ] || { log "$d 없음 — 멈춤"; exit 1; }; done
# ② 는 w10m 평가가 다 끝난 뒤 (사진 특징 계산이 GPU 를 꽉 쓴다)
until tail -1 outputs/piper/world10m.log | grep -qE "끝$|멈춤"; do sleep 120; done
log "[② 새 데이터 사진 특징 + 위치 입력] $(for d in $NEW; do echo -n "$(basename $d) $(ls $d/episodes/*.npz | wc -l) "; done)"
VP=()
for s in 0 1; do
  ( $PY vis_cache.py build --policy outputs/piper_np3_model/merged --data $NEW --out cache/vis --batch 8 --stride 6 --shard $s --nshard 2 2>&1 | grep -E "끝|Error|Traceback" ) &
  VP+=($!)
done
wf2c(){ local FP=(); for s in 0 1 2; do ( PIPER_WORLD=piper_sim/worlds/$2.json nice -n 10 $PY piper_sim/world_feat_cache.py --data $1 --shard $s --nshard 3 --stride 6 --version 2 --sure --tag c 2>&1 | grep -E "^끝|Traceback|Error" ) & FP+=($!); done; wait "${FP[@]}"; }
for c in default random many; do wf2c data/world_motion2_$c $c; wf2c data/world_daggerW1_$c $c; wf2c data/world_daggerW2_$c $c; wf2c data/world_daggerW3_$c $c; done
wait "${VP[@]}"
D45="data/world_dagger4_default data/world_dagger4_random data/world_dagger4_many data/world_dagger5_default data/world_dagger5_random data/world_dagger5_many"
DN="data/world_daggerN_default data/world_daggerN_random data/world_daggerN_many"
D6="data/world_dagger6_default data/world_dagger6_random data/world_dagger6_many"
N10="data/world_motion_default data/world_motion_random data/world_motion_many data/world_daggerM2_default data/world_daggerM2_random data/world_daggerM2_many
     data/world_daggerM3_default data/world_daggerM3_random data/world_daggerM3_many data/world_daggerK3_default"
DATA="data/world_default_t data/world_random data/world_many data/world_dagger_d1 data/world_dagger_r1 data/world_dagger2_default data/world_dagger2_random data/world_dagger2_many data/world_dagger3_default data/world_dagger3_random data/world_dagger3_many $D45 $DN $D6 $N10 $NEW"
PIPER_WORLD=piper_sim/worlds/default.json $PY piper_sim/world_feat_cache.py --data $NEW --version 3 --from-v2 --sure --tag c | tail -1
for d in $DATA; do
  a=$(ls $d/episodes/*.wfeat3sc.npy | wc -l); b=$(ls $d/episodes/*.npz | wc -l)
  [ "$a" -eq "$b" ] || { log "$d wfeat3sc $a / $b 모자람 — 멈춤"; exit 1; }
done

log "[③ w11m 학습] $CD 에서 15000스텝, + 새 데이터 (w10m DAgger·시범), 지시문 바꿔 쓰기 50% (MemAvailable $(mem_av)GB)"
TR="--full-expert --balance --workers 0 --rtc-max-delay 14 --ema 0.999 --log-every 500 --abs-pos --objfeat --feat wfeat3sc --renorm-new-feat --renorm-action --vis-cache cache/vis --frame-stride 6 --sufstate --fp32-master --vlm-lora 16 --vlm-lora-lr 2e-4 --proj-lr 3e-4 --new-lr 3e-4"
PIPER_WORLD=piper_sim/worlds/default.json $PY train_lora.py --policy $CD --data $DATA $TR --instr-aug 0.5 --batch-size 8 --grad-accum 2 --steps 15000 --lr 3e-5 --eval-every 5000 --save-every 5000 \
    --out outputs/piper_w11m_model 2>&1 | grep --line-buffered -E '에피소드 학습|새로 쓰는|"step": [0-9]*000, "loss"|val_loss|모델 저장|Traceback|Error|Killed'
CW=outputs/piper_w11m_model/merged
[ -f $CW/worldfeat_sure ] && [ -f $CW/worldfeat_v3 ] && [ -f $CW/worldfeat_cammask ] || { log "w11m 모델 표시 없음 — 멈춤"; exit 1; }

[ -s outputs/piper/ra_choice.txt ] && RA=$(cat outputs/piper/ra_choice.txt)   # w10m 에서 amb 를 견준 뒤 정한 다듬기 부품
log "[④ w11m 새 기준 평가] 다듬기 $RA (GPU 일꾼 넷)"
E4="ev:default:1:1000-1049:$RA:w11mraM:$CW ev:random:1:1000-1049:$RA:w11mraM:$CW ev:many:1:1000-1049:$RA:w11mraM:$CW
    ev:default:1:1000-1049:-:w11mM:$CW ev:random:1:1000-1049:-:w11mM:$CW ev:many:1:1000-1049:-:w11mM:$CW
    ev:default:2:1000-1019:$RA:w11mraM:$CW ev:random:2:1000-1019:$RA:w11mraM:$CW ev:many:2:1000-1019:$RA:w11mraM:$CW
    ev:default:3:1000-1019:$RA:w11mraM:$CW ev:random:3:1000-1019:$RA:w11mraM:$CW ev:many:3:1000-1019:$RA:w11mraM:$CW"
W=()
for i in 1 2 3 4; do gworker $E4 & W+=($!); sleep 5; done
wait "${W[@]}"
grep -h SUMMARY outputs/piper/w11m_w11m*_k*.log
for c in default random many; do PIPER_WORLD=piper_sim/worlds/$c.json $PY piper_sim/world_table.py outputs/world_w10mM_$c outputs/world_w10mraM_$c outputs/world_w11mM_$c outputs/world_w11mraM_$c; done
log "끝"
