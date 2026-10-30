# 인계 메모 — 10/30 22:50 절이 최신

## 10/30 22:50 (가장 최신, 이것부터)

- w10m 학습 (piper-w10m ③): 5000스텝 검증 손실 0.0896 → 10000스텝 0.0847 (w9 의 0.0806 과는 검증 묶음이 달라 바로 못 견줌). 23:30쯤 끝나면 ④ 평가 12개가 저절로.
  그 뒤 piper-w10mamb (같은 장면 RA+amb, outputs/world_w10mrabM_*).
- 새 기준 w9 + 다 켬 일 2개 0/20 을 일별로: 첫 일 낯선 꼴 1/11·쌓기 0/5·익숙 2/4, 둘째 일 낯선 4/13·쌓기 0/4·익숙 1/3 → 일 1개와 같은 원인 (지시문 꼴·쌓기).
  스크립트 scratchpad/motion/k2_fail.py. 새 기준 일 1개 실패 중 '지시한 물체 안 쥠' 은 대부분 아무것도 안 쥔 경우 (다른 물체를 쥔 것은 많음 설정에서 8번).
- world_dagger.py (일 1개 DAgger) 를 새 기준으로: sample_motion (지연·속도·잡음) + pad_lang + --fix-wrong (다른 물체를 쥐고 있으면 손 가까운 빈 자리에
  내려놓고 지시한 물체로, 기록). 백업 outputs/piper/world_dagger_before_motion.py. 아직 GPU 로 안 돌려 봄 (학습 중이라).
- world_dagger_nested·world_dagger_deep 에 --ret-p (B 를 마친 뒤 그 비율만큼 VLA 가 돌아오기를 먼저 T2 스텝 해 보고 시범이 이어받음, 기록)
  + 돌아오다 다른 물체를 쥐었으면 내려놓기. 백업 outputs/piper/world_dagger_nested_before_ret.py, world_dagger_deep_before_ret.py. 역시 GPU 로 아직 안 돌림.
- 22:37 piper-co11 (run_collect_w11.sh): ④ 시작 10분 뒤 시범 세 설정 × 80 (24000~, PIPER_STACK_P=0.5) → data/world_motion2_*.
- 22:45 piper-w11m (run_world11m.sh, 기록 outputs/piper/world11m.log): ④ 시작 15분 뒤부터 w10m 으로 DAgger (일 1개 --fix-wrong 21000~ ×100,
  깊이 2 --ret-p 0.6 22000~ ×60, 깊이 3 23000~ ×20 — 세 설정, w10m 평가와 자물쇠를 나눔) → co11 기다림 → w10m 파이프라인 '끝' 뒤 ② → ③ w11m (w10m 에서 15000,
  train_override obj_frac 0.15) → ④ (다듬기는 outputs/piper/ra_choice.txt 가 있으면 그 내용 — amb 비교 뒤 써 둘 것). claim 폴더 outputs/piper/claim_w11m.
  w10m 이 w9 보다 나쁘면: systemctl --user stop piper-w11m → claim_w11m 지우고 CD=outputs/piper_w9_model/merged 로 다시.
  첫 DAgger 항목 로그 (outputs/piper/w11m_d1_*.log) 에서 '예외' 가 줄줄이 나오면 바로 멈추고 고칠 것.

## 10/30 11:20

- w9 결과 (판 2개, 예전 기준): 일 1개 3cm 혼자 34·28·32%, 예전 다듬기 74·50·66%, 다 켬(w9ra) 88%·(흔든 28장면 46%)·74%. 일 2개 35·5·10%, 일 3개 10·0·5%.
- 사용자 10:50~11:10: '판도 n개로 해서 하는거 맞지?' → 아니었다 (평가가 판을 max(2,k) 로 고정, 설정 n_pads 를 안 읽음). '동작을 여러 변수를 추가해줘'
  (골라 줌: 끼어드는 때·일 종류·지시문·지연·속도 + '더 좋은 게 있으면 추가'), '지금 하드코딩으로 하는건가?', '동작 하나당 몇 스텝?' (시범 단계표 답함, VLA 는 재는 중).
- 고친 것 (백업 outputs/piper/*_before_npads.py, *_before_motion.py, world_refine_before_pfine2.py, fastinfer_before_lru.py, worlds/*_before_motion.json):
  · world.pick_n_pads: 판 개수 = 설정 n_pads 범위 (일 개수 이상). world_eval --n-pads 로 예전처럼 고정 가능. DAgger 셋도 설정에서.
  · 설정 motion (세 설정 모두 켬): switch.phase_p (reach 0.2·grasp 0.2·carry 0.4·place 0.2, 값 범위 reach_cm 3~10·grasp_steps 1~10·carry_steps 20~80·place_cm 3~8, fallback),
    task_p (쌓기 25%), latency_steps 6~12, speed 0.8~1.2, action_noise [0.015, 0.0075], friction 0.6~1.4, grip_friction 0.6~1.0, mass 0.5~2.0.
    instruction 틀: 판 5개·쌓기 4개 (첫 틀 = 예전). PIPER_MOTION=0 이면 모두 끔 (예전 장면 그대로 — 단계 측정에서 확인 중).
  · world: templates/instruction(rng)/rephrase/motion/sample_motion/vary_physics(place 안에서). chain 이 task_p·틀을 씀.
  · world_eval: make_trigger (네 가지 때), fallback (때가 안 오면 끝난 뒤 다음 지시), 장면마다 지연, release_wait 에도 다듬기 (--no-release-refine 로 예전),
    일마다 events [near, held, lift, arrive, release] + start_step + switch, 결과에 motion. pad_lang (지시문 48토큰 채움 — CUDA 그래프 하나).
  · switch_experiment.Episode.step: motion 의 speed·noise (기록은 명령 그대로). fastinfer: 그래프가 넘치면 가장 오래 안 쓴 것을 지움 (EVICT).
  · world_expert: approach_until·carry_for·carry_until·to_switch, run_chain(switch=). world_collect·DAgger 둘이 sample_motion 을 씀.
  · train_lora: --instr-aug p (지시문 틀 바꿔 쓰기), --balance 는 canon(같은 일) 기준으로 셈 (틀마다 따로 세면 새 데이터가 몇 배로 뽑힘).
  · world_refine pfine2: 막대만 든 물체 가운데 어긋남 허용 (긴 축, 3.5cm 까지 — 흔든 1006·1019 는 조명 탓에 든 물체를 다른 물체로 읽어 4cm 틀림),
    1.2cm 위까지 내린 뒤 천천히 내려 닿아 멈추면 폄 (흔든 1023 은 몇 cm 위에서 놓아 굴렀다). VLA 없는 놓기 시험 3/3 (0.3·0.4·1.1cm).
- 유닛: piper-npad (run_npad_base.sh: 판 N개·동작 변수 없음 w9 혼자 기본·흔든 50 → outputs/world_w9P_*, 깊이 3 DAgger 기본 남은 장면 — 많음은 11:15 에 PID 로 끔),
  piper-phase (VLA 단계 측정: PIPER_MOTION=0 --n-pads 2 --no-release-refine 로 예전 w9ra 기본 1000~1011 → scratchpad/phase), piper-mtest (시범 끼어드는 때 넷 시험, 5/5 성공 중).
  piper-w10 (run_world10.sh) 은 10:57 에 멈춤 (예전 고정 끼어들기로 DAgger 를 모으게 되어 있어서).
- 11:39 piper-w10m (run_world10m.sh, 기록 outputs/piper/world10m.log) 띄움: ⓪ w9 새 기준 (world_w9raM_*·world_w9M_* k1, w9raM 기본 k2·k3)
  ‖ ① DAgger (data/world_daggerM2_*·M3_*) ‖ 시범 CPU 2줄 (data/world_motion_*) → ② 특징 → ③ w10m (--instr-aug 0.5) → ④ w10mM·w10mraM. 일꾼은 claim 폴더
  outputs/piper/claim_w10m 로 항목을 나눔 (다시 띄우면 지우고 처음부터 — 평가·DAgger·수집은 끝난 장면을 건너뜀).
- piper-gif (run_gif_lock.sh: 자물쇠 거쳐 world_w9ra_s1024 4K, PIPER_MOTION=0 판 2개 — 일지 10/30 에 넣을 것).
- 12:03 깊이 3 DAgger 기본 (예전 고정 끼어들기, piper-npad) 을 PID 로 끔 (15030~15039 남음) — 새 기준 GPU 작업 셋 + 이것 + 수집 EGL 0.5GB 로 GPU 여유가 0.2GB 까지.
  판 N개 w9 혼자: 기본 23/50, 흔든 18/50 (outputs/world_w9P_*). piper-npad 끝.
- 14:32 새 기준 w9 혼자 기본 9/50 (18%, 판 안 15) — 같은 장면 다 켬 27/50 (다듬기만 성공 20, 혼자만 2). 판 일 익숙한 꼴: 혼자 4/13 → 다 켬 13/13.
  동작 변수(지연·속도·잡음·마찰)는 VLA 혼자에게 크게 불리하고 다듬기가 거의 다 메움 (논문: 보조의 몫). 흔든 다 켬 24/50, 익숙한 꼴 13/13, 쌓기 0/12.
- 13:18 새 기준 w9 + 다 켬 기본 50장면 27/50 (54%, 판 안 28): 쌓기 0/12 (모두 지시한 물체를 안 쥠), 판 일 27/38 (71%).
  판 일만: 지시문 put/place 꼴 13/13, move 9/13, set…down 3/7, pick up…and put 2/5 · 지연 11~12 59% (6~10 81%) · 속도 1.07~ 62% · 손가락 마찰 낮음 64%.
  분석 스크립트: scratchpad/motion/by_var.py, fail_kind.py.
- 12:35 새 기준 w9 + 다 켬 기본 19장면 10/19: 쌓기 0/4 (VLA 가 지시한 물체를 아예 안 쥠 — 쌓기·새 지시문 꼴을 못 배움), 판 10/15.
  → train_lora --obj-frac (balance 와 함께 쌓기 일 몫), outputs/train_override.json 에 piper_w10m_model: obj_frac 0.15 (파이프라인은 그대로, 학습 때 덮어씀).
- 20:14 ③ w10m 학습 시작 (23:20쯤 끝). balance.json: 쌓기 몫 46.8% → 15% (쌓기 일 211가지·프레임 10087/424448), 지시문 바꿔 쓰기 0.5. 학습 13185구간.
  w9 일 3개 새 기준(2장면에서 멈춤)은 ④ 뒤 GPU 가 비면 이어서.
- 19:00 ② 진행: vis_cache 2줄 (GPU 100%, 665/1647 → 20:15쯤 끝). run_world10m.sh 의 wf2c 안 'wait' 가 인자가 없어 vis_cache 까지 기다림 → 위치 입력이
  첫 폴더 뒤로 밀림 (고장은 아님). 남은 9폴더는 piper-wf2c (scratchpad/wf2c_rest.sh) 로 미리 계산 — 파이프라인은 있는 파일을 건너뜀.
  다음 파이프라인을 만들 때 wf2c 의 wait 를 그 셋의 PID 로 고칠 것 (run_world10.sh 도 같은 버릇). train_lora 가 balance.json (쌓기 몫) 을 모델 폴더에 남김.
- 18:00 새 기준 w9 + 다 켬 일 2개 기본 0/20 (일별 8/40; 끼어드는 때 reach 7·grasp 3·carry 9·place 1, 실제로 끼어듦 13/20).
  w9 일 3개 새 기준 평가는 2장면에서 PID 로 멈춤 (w10m 학습을 2시간 앞당기려고) — 나중에 이어서 재기 (끝난 장면은 건너뜀).
- 17:10 새 기준 w9 혼자 많음 7/50 (다 켬 22, 다듬기만 15·혼자만 0). 1044·1048 은 혼자일 때 지시한 물체를 쥠 → 다듬기가 옆 물체로 끌고 감 (amb 대상).
  piper-w10mamb (run_w10m_amb.sh): 파이프라인이 '끝' 을 찍으면 w10m + RA+amb 일 1개 세 설정 (outputs/world_w10mrabM_*). 일지 6절에 새 기준 표 (e9ed15a).
- 16:20 쌓기 수집 2줄 끝 (기본 47/50·많음 44/50). 새 데이터 1429구간 중 쌓기 323. 새 기준 w9 혼자: 기본 9/50·흔든 10/50 (다 켬 27·24).
- 15:50 다듬기 새 부품 "amb" (기본 꺼짐): 손 아래 두 물체가 1.5cm 안으로 비슷하면 쥐기 다듬기를 안 붙임. 근거: 새 기준 w9 기본·흔든에서 '지시한 것은 안 쥐고
  다른 것만 쥠' 이 다 켬 때만 5번 (혼자 0), 흔든 1014 는 혼자 성공·다 켬 실패 (다듬기가 옆 파란 정육면체에 붙음), 많음 1044·1048 도 다듬기가 처음 붙은 물체 = 잘못 쥔 물체.
  할 일: w10m 평가 뒤 같은 장면에서 RA 와 RA+amb 비교 (outputs/world_w10mrabM_*).
- 15:34 새 기준 w9 + 다 켬 일 1개 세 설정 끝: 73/150 (49%) — 기본 27·흔든 24·많음 22 /50. 판 일 익숙한 꼴 34/38 (89%), 낯선 꼴 39/79 (49%), 쌓기 0/33.
  많음 익숙한 꼴 실패 4 중 3 은 다른 물체를 쥠 (같은 모양 다른 색 둘) — VLA 의 물체 고르기. 판 N개 DAgger 깊이 2: 기본 42/60, 흔든 43/60.
- 15:10 시범 수집 6줄 끝 (300장면 278 성공, 883구간, 쌓기 148구간). 쌓기를 더: piper-costack-default/-many (PIPER_STACK_P=0.7 → 실제 쌓기 42%,
  seeds 20600~20649 / 20800~20849, 같은 data/world_motion_* 폴더에 — 파이프라인 ② 가 그대로 특징 계산). world.motion() 에 PIPER_STACK_P 덮어쓰기 추가.
- 14:57 GPU 문지기를 piper-gpugate2 로 다시 띄움: python 만 센다 (RustDesk 원격 접속이 GPU 계산 프로세스로 잡혀 python 2개만 돌았음).
  GIF world_w9ra_s1024 (4K, 성공) 일지에 넣음 (0f48f6a).
- 12:04 piper-gpugate (run_gpu_gate.sh): 시범 수집(world_collect)이 도는 동안 GPU 작업이 3개 이상이면 GPU 자물쇠를 대신 쥠 → 3개까지. 수집이 끝나면 스스로 끝.
  (수집 줄은 MuJoCo 그림 그리기로 GPU 0.25GB씩 — 4개 + 수집 둘이면 11.0GB/11.26GB)
- 판 N개 VLA 혼자 중간 (11:44): 기본 18/39, 흔든 12/33 — 판 2개(34%·28%)보다 낮지 않음 (장면이 달라 짝 비교 아님). 판 많을수록 조금 낮아 보임.
- 일지 10/30 초안 push (124916f). GIF·판 N개 결과·새 기준 w9 결과를 채울 것.


## 10/30 00:50 (가장 최신, 이것부터)

- 유닛: piper-w9 (위치 입력 계산 → 01:40쯤 학습 → 04:40쯤 w9 평가), piper-idle9d (w8rmg·deep3 시험, '[②] 끝' 이면 끔),
  piper-idlera (run_idle_ra.sh: w8 + 다듬기 다 켬 일 1개 물체 많음→기본→흔든 → outputs/world_w8ra_*, 같은 조건에 끔, 다시 띄우면 이어서),
  piper-w9side (run_w9_side.sh: w9 일 1개 평가 단계 동안만 w9ra, 일 2개 단계로 넘어가면 끄고 그만),
  piper-w10wait (run_w10_wait.sh: piper-w9·piper-w9side 가 끝나면 run_world10.sh 를 piper-w10 유닛으로).
- run_world10.sh: ⓪ w9r 과 w8rm 중 나은 모델 C, w9ra 남은 장면 3 + 깊이 3 DAgger 기본 (15000~15039) ① 끼어들기 DAgger 깊이 2 세 설정 × 120 (14000~, data/world_daggerN2_*)
  → 깊이 3 흔든·많음 (15100~, 15200~, data/world_daggerK3_*) ② 사진 특징·wfeat2sc·wfeat3sc ③ w10 학습 (C 에서 15000, + 새 DAgger)
  ④ w10 + 다 켬 일 1개 150, 예전 다듬기·혼자 150, 다 켬 일 2개·3개 20×3. GPU 는 DAgger·끼어들기 평가 3개까지 (2.9GB씩).

## 10/30 00:45

- gplan 빈손 원인 (scratchpad/yawt/diag_gplan.py, 기본 1024·1039): 위 카메라 막대 긴 축 기억(ymem)이 90° 틀림 (처음 사진은 2.5° 안 → 팔이 다가와 가릴 때 뒤집힘)
  → 손가락이 막대 길이 방향으로 벌어져 윗면에 얹힘. 고침 ① world_gplan: 내려가기 전 손목 사진 방향 확인 (fmask 켰을 때, 12° 넘으면 ymem 을 고쳐 계획 다시 = gplan_wfix)
  ② world_refine._update: 막대 ymem 은 기억과 30° 넘게 다르고 길쭉함 < 1.8 이면 안 고침.
- pfine 이 물체를 판에 닿게 내리면 평가(Track)가 '든 채 성공' → release_wait 는 다듬기 없이 VLA 만 → 물체를 다시 듦 (z 3~4cm 로 끝).
  → pfine 은 판 윗면 1.2cm 위에서 편다. (world_eval 의 release_wait 에 다듬기가 빠진 것은 그대로 둠 — w8 비교 때문)
- 다 켬 (grasp,place,fmask,settle,wxy,gplan,objc,pfine) 빠른 시험 w8 기본 1036·1039·1024: 3/3 성공 (w8rm 은 셋 다 실패). 고른 장면이라 효과는 w9 150장면으로.
- 실수: 시험 프로세스를 pkill -f 로 끄려다 막힘 → PID 로 끔. 규칙대로 PID·systemctl 만.

## 10/30 00:10

- 23:20 절 뒤에 더한 것: pfine (놓기 마무리 — 판 위에서 펴려 하면 든 물체 가운데를 판 가운데 0.8cm·판 윗면 0.8cm 안으로 맞춘 뒤 폄),
  gplan follow (내려가기 전까지 기억 자리를 따라감, 백업 outputs/piper/world_gplan_before_follow.py),
  시범 판 비우기 world_expert.blockers/clear_dest → world_dagger_nested·world_dagger_deep 가 B(마지막)·돌아오기 놓기 전에 판 위 다른 물체를 빈 자리로
  (백업 outputs/piper/{world_expert,world_dagger_nested,world_dagger_deep}_before_cleardest.py, VLA 없는 시험 3/3, STATS 에 cleared).
- VLA 없는 시험 (scratchpad): wxy 기억 3.5cm 틀린 막대 켬 12/12·끔 7/12, pfine+objc 막대 2.5cm 치우쳐 쥠 4/4·끔 3/4, 다 켬(all) 막대 쥠 4/4 (gplan 첫 시도는 틀린 기억으로 빈손 → follow 로 고침).
- `piper-w9side` (run_w9_side.sh): w9 일 1개 평가(⑤ 앞 두 단계) 동안만 GPU 4번째 자리에서 w9 + grasp,place,fmask,settle,wxy,gplan,objc,pfine
  기본→물체 많음→흔든 (outputs/world_w9ra_*). 파이프라인이 일 2개 평가로 넘어가면 끄고, 파이프라인이 끝나면 이어서. 기록 outputs/piper/w9_side.log
- piper-idlefs 는 00:06 에 멈춤 (CPU 부하 40 으로 위치 입력 계산이 느려져서). w8rfs 기본 35장면 23 → 24.
- w8rmg (gplan) 00:06: 기본 35장면 23 → 24 (+2/−1), 물체 많음 28장면 15 → 20 (+5/−0).
- 일지 10/29 7절 (밤) push (db8582e).

## 10/29 23:20

- 돌고 있는 것: `piper-w9` (② 위치 입력 다시 계산 — dagger3~N 남음, 00:30쯤 끝 → wfeat3sc → w9 학습 ~3시간 → w9 평가),
  `piper-idle9d` (run_idle_w9.sh: w8rmg = w8 + gplan 기본·물체 많음, diag_bar/diag_place 추적, 깊이 3 DAgger 시험 — '[②] 끝' 이면 스스로 끔),
  `piper-idlefs` (run_idle_fs.sh: w8 + grasp,place,fmask,settle 일 1개 기본→흔든→많음 → outputs/world_w8rfs_*, 같은 조건에 스스로 끔. 다시 띄우면 이어서).
  끄기는 PID 나 systemctl --user stop 만.
- **막대 못 쥠 원인 (diag_bar 추적, 손목 사진 scratchpad/yawt/diag_wrist.py → yawt/wrist_*)**:
  ① 검은 막대가 손목 사진에서 검은 그리퍼 손가락과 한 덩어리 → 손목 맞추기가 −90° 로 읽고 반대로 돌리며 내려가기를 막음 (기본 1004)
  ② VLA 가 쥐는 높이(0.8cm)·방향 0.8° 로 내려와 오므리며 같은 스텝에 손을 듦 → 그리퍼(다 닫히는 데 5~6스텝)가 허공에서 닫힘 (기본 1024).
     grasp_assist._descend 의 ⑩' (close=False 일 때 VLA 가 오므리면 dmode 끄고 VLA 동작 그대로) 탓
  ③ 다듬기 기억 자리가 3.3~4.3cm 틀림 (위 카메라가 팔에 가려 덜 보일 때 잰 값, 손 8cm 안이면 안 고침) → 엉뚱한 곳에 맞춤 (기본 1022·1024, 흔든 1015·1022)
  ④ VLA 가 옆으로 미는데 다듬기가 맞추기·오므림 미루기로 버팀 → 1cm 어긋난 채 교착, VLA 가 포기 (기본 1036)
  ⑤ 다듬기가 붙으면 못 쥠: 150장면 중 혼자는 20스텝+ 쥐었는데 다듬기는 0 인 장면 10, 반대 18 (scratchpad 계산)
- **새 다듬기 부품 (기본 꺼짐, w9 평가 영향 없음, 백업 outputs/piper/{world_detect,world_assist,world_refine}_before_fingermask.py)**:
  · fmask: world_detect.WristFingerMask (그리퍼 XML 손가락 메시 + 손목 카메라 투영, 그리퍼 값만 봄, 저장 사진과 겹침 99%)로 손가락 자리를 지우고
    기억 자리 투영에 가장 가까운 덩어리로 방향을 잰다. 저장 사진 24장: 예전 −90° 18장 → 실제와 2~4°.
  · settle: 쥐는 높이 1.5cm 안·옆 1.5cm 안에서 오므리기 시작하면 그리퍼가 다 닫힐 때까지 (최대 10스텝) 손을 붙잡음.
  · wxy: 손목 사진 덩어리를 윗면 높이 평면으로 되쏘아 기억 자리를 반씩 고침 (카메라 자리는 로봇 상태 eef + 그리퍼 XML, 시뮬 카메라와 0.00cm).
    VLA 없는 시험: 원기둥 옆 2.8cm 까지 0.2cm (중앙). 막대는 사진 가장자리에서 잘리면 긴 축 선으로 짧은 축 방향만 고침.
  · 빠른 시험 (기본 1004·1024·1036, fmask+settle): 1024·1036 은 막대를 쥠 (예전 0) — 그 뒤 VLA 가 다른 곳에 놓아 실패. 1004 는 손목 관절 한계(unjam).
- gplan (w8rmg, 23:19): 기본 18장면 13 → 13, 물체 많음 13장면 6 → 9 (얻음 3, 잃음 0). 개입률 25% (원래 18%).
- 일 2개 (w8r 60장면): VLA 가 A 를 B 의 판 위에 내려놓은 장면 22 (scratchpad/k2_putdown.py). 'B 를 쥐었지만 3cm 밖' 은 대부분 B 를 판 3cm 안에 놓은 뒤
  A 를 B 판에 놓으며 밀어냄 (기본 1008·1014·1015). 시범 프로그램은 판을 피해 빈 자리를 고른다 → 끼어들기 DAgger 를 더 (다듬기는 지시문을 안 읽으니 못 막음).


## 10/29 19:30

- w8 혼자 + 고친 가리개 (outputs/world_w8m_*): 3cm 36/150, 판 안 52/150 (예전 34, 48). 일지 올림 (c3d390c).
- 다듬기 새 선택 항목 (기본 꺼짐, w9 평가에는 영향 없음 — 백업 outputs/piper/world_refine_before_objc.py, _before_gplan.py):
  · objc: 놓기 다듬기가 손 대신 든 물체 가운데를 판 가운데로 (--refine grasp,place,objc)
  · gplan: 쥐는 자세 계산 piper_sim/world_gplan.py (위 카메라 자리·긴 축 방향 + 역기구학, 책상 같은 고정물만 충돌 확인) (--refine grasp,place,gplan)
- VLA 없는 시험 (scratchpad/yawt): 손목 사진 막대 방향 재기는 25° 안에서 2° 안으로 맞음. 경로 계산으로 막대 위 6cm 에 δ±30°·옆 1.5cm 로 둔 뒤
  똑바로 내려 쥐면 다듬기 켬 7/7, 끔 4/6 → 지금 다듬기는 이 조건에선 쥔다. 실제 실패는 VLA 의 움직임과 얽힌 것 (가설).
- piper-idle9b 유닛 (run_idle_w9.sh): piper-w9 가 '[②] 끝나기 기다림' 에 오면 GPU 3개로 ① w8 + objc 기본·흔든 (outputs/world_w8rmc_*)
  ② 막대 실패 12장면 추적 (piper_sim/diag_bar.py → outputs/piper/diag_bar_*.json, 요약 scratchpad/yawt/diag_sum.py) → 깊이 3 DAgger 시험
  (data/world_daggerK3_test). 위치 입력 계산이 끝나면 ('[②] 끝') 학습 전에 스스로 끈다. 기록 outputs/piper/idle_w9.log
- DAgger 6차 이어받기 실패 15 중 12 가 VLA 가 넘어뜨린 목표 (기울기 90°, 모양 가리지 않음, 전체 65장면의 18%) → 시범 프로그램이 넘어진 물체도 쥐게
  (world_expert.toppled_candidates: 지금 자세 겉 점으로 높이·수평 폭, 20° 넘게 기울었을 때만; place 는 가장 낮은 점 기준. 백업 outputs/piper/world_expert_before_topple.py).
  평가 기록에 tilt_max_deg (쥐지 않은 동안 목표가 가장 많이 기운 각도) 추가 — 동작은 그대로 (백업 world_eval_before_tilt.py).
- 표: '지시한 물체로 감' 을 혼자·다듬기 같은 기준(손 3cm 안·쥠)으로, 다듬기 first_intent 는 따로 (world_table.py).
- VLA 없는 시험 (scratchpad/yawt, CPU osmesa): 넘어진 물체 시범 (topple_test.py) 막대·정육면체·삼각기둥 5/5 쥐어 판에 놓음 (원기둥·육각 기다림).
  깊이 3 장면 13000~13002 시범 프로그램 혼자 3/3 (expert_deep.py) → VLA 와 한 DAgger 시험 실패는 VLA 가 남긴 상태 탓.
  막대 쥐기 (planned_grasp.py): 지금 다듬기 기본 10/10·물체 많음 7/8, gplan 10/10·7/8 (그 조건에선 차이 없음).
  넘어진 물체 시범: 5모양 × 3 = 15/15 쥐어 판에 놓음. 일지 올림 (d36fdcd).
- 평가는 같은 장면·같은 입력이면 그대로 재현된다 (w8r ↔ w8rm 스텝 수 같은 장면 31/50·36/50) → 부품 하나 바꾼 비교는 그 부품이 켜진 장면만 달라진다.
  짝 비교: scratchpad/cmp_runs.py <A> <B> [설정…] (outputs/world_<A>_<설정>/k1n.json). 넘어뜨림 세기: scratchpad/tilt_sum.py <폴더…>

## 10/29 18:40

- piper-w9 는 그대로 도는 중 (① 혼자 평가 → ③ DAgger 6차 → ② 위치 입력 다시 계산 기다림 → ④ w9 학습 → ⑤ 평가). 위치 입력 다시 계산(CPU)이 가장 느리다: 18시에 world_many 2470/2924, 전체 끝은 10/30 02시쯤 예상.
- 고친 가리개 결과 (w8 + 다듬기, 일 1개 150장면, outputs/world_w8rm_*): 3cm 85 → 87, 판 안 92 → 93. 검정 4 → 11/22, 갈색 9 → 12/22, 나머지 6색 72 → 64/106 (예전만 12, 고친 뒤만 4). 일지 10/29 4절에 적고 올림 (7b3a44a).
- 학습 데이터로 확인 (scratchpad/maskcmp.py): 다른 색 위치 입력은 나빠지지 않음 (손 5cm 안 5cm 넘게 틀림 7.5% → 5.6%). '보임' 칸만 2~6%p 줄어듦. w9 에서 다시 본다.
- w8rm 실패 60 중 쥐기 관련 32 (못 쥠 13, 다른 물체 쥠 10, 금방 놓침 6, 못 다가감 3), 그중 막대가 절반. '다른 물체 쥠' 은 처음 자리가 붙어 있던 것이 아님 (scratchpad/nbr.py) → 목표를 못 쥔 뒤 VLA 가 딴 물체로 감.
- 다음 생각 (아직 안 함, 가설): 다듬기에 '쥐는 자세 계산' 부품을 새 이름으로 더하기 (기본값은 그대로 두어 w9 비교를 안 흐림). 위 카메라로 잰 위치·방향 + 로봇 모델 역기구학(task_experts._grasp_bowl_piper 처럼 기울기·대칭 후보)으로 쥐는 자세를 정해 내려가 쥐고 조금 든 뒤 VLA 에 돌려줌. 옆 물체 충돌은 잰 위치 + 설정 크기로 (시뮬 정답 쓰지 않기). 놓기는 손이 아니라 든 물체 가운데를 판 가운데로.
- 시험은 GPU 가 빌 때만 (DAgger 6차 3개가 도는 동안 4번째는 nvidia-smi 로 3GB 넘게 남을 때만). 끄기는 PID 나 systemctl --user stop 으로만.

## 10/29 16:10
- 사용자 (15:45): "95퍼센트 될때까지 내가 말한것들 다 기억하지? 연구 진행하자" + "연구장면도 2개만 옮기는 게 아니라 n개 다 옮길 수 있게" → 그만하라 할 때까지 계속, 목표에 끼어들기 깊이 N (메모리 research-scenario·world-goal 갱신)
- w8r 일 1개 실패 65 (scratchpad fail8.py): 다가갔지만 못 쥠 20 (검정 9·갈색 6), 다른 물체 쥠 8, 판 안 3cm 밖 7, 쥐었다 금방 놓침 7, 다른 판 7 … 막대 12/38, 검정 5/22, 갈색 9/22
- 일 2개 (w8r 60장면) 처음 막힌 곳: 둘 다 10, B 쥐었지만 3cm 밖 19, A 30스텝 못 쥠 17, B 못 쥠 8, 돌아오기 실패 6 (그림 06_결과물/PiPER/img/world_w8_k2_breakdown.png). 4K 성공 장면 many 1017 은 A 를 자기 판에 바로 내려놓음 (빈 자리 아님) → 성공 장면 중 몇 개가 이런지 세기
- **검은 물체 원인 (scratchpad trace/trace_feat.py, 기본 1000)**: 쥐기 직전 (손-막대 1.5cm) VLA 물체 기억이 10.5cm 튐 — 그리퍼 옆 손목 카메라 몸체가 '검은 물체'로 잡히고 손에서 11cm 라 FAR(10cm) 규칙으로 기억을 덮음. RobotMask 가 link·base 몸체 + 손끝 점만 가리고 10/24 의 wrist_cam geom (right_hand) 은 안 가렸음
- **고침 (world_detect.RobotMask)**: self.extra = wrist_cam* geom (상자 모서리 8 + 가운데), install 없이 부를 때도 make_cam_robot() 모델 (world_feat_cache 용). 백업 outputs/piper/world_detect_before_cammask.py. 같은 장면 성공 (0.7cm), 손 5cm 안 기억 오차 0.1~1.9cm. 흔든 환경 1017·1024 는 첫 사진부터 검정을 못 찾거나 갈색으로 (그리퍼 옆, 색 흔들기) — 아직
- world_feat_cache --tag (c → *.wfeat2sc / *.wfeat3sc, 예전 파일 그대로), train_lora --feat wfeat3sc → 표시 worldfeat_cammask (백업 outputs/piper/{world_feat_cache,train_lora}_before_cammask.py)
- **`piper-w9` (run_world9.sh, log outputs/piper/world9.log, MemoryMax 24G, 16:01 시작)**: ① w8 + 고친 가리개 일 1개 다듬기 (world_w8rm_{cfg}) → 혼자 (world_w8m_{cfg}) ‖ ② CPU wfeat2sc 다시 (log world9_feat.log) ③ DAgger 6차 (w8, 12000~12599, data/world_dagger6_{cfg}) → 사진 특징·wfeat2sc ④ wfeat3sc → w9 학습 (w8 에서 15000) ⑤ w9 일 1개 다듬기·혼자, 일 2개·일 3개 다듬기 20장면 ⑥ 표. 내일 아침 6시쯤 끝 예상
- 일지 10/29 push (fc3b858). 주간보고 10/29 (쉬운 말 슬라이드, 대본 8문장, 예상 질문 14) 와 저장소 정리도 오늘

## 10/29 14:40
- `piper-w8` 09:34 끝 (GPU 비어 있음, 돌고 있는 것 없음). **w8 (w7 에서 15000, + 끼어들기 DAgger, val 0.0877)**: 혼자 12·12·10 = 34/150 (23%)·판 안 48, 다듬기 31·26·28 = **85/150 (57%)**·판 안 92 (개입 18.5%). 일 2개 다듬기: default 5/20 · random 1/20 · many 4/20 (w7r default 6/20) → 끼어들기 DAgger 로 일 2개는 오르지 않음. 표 outputs/piper/world8.log 끝. **아직 일지에 안 씀**
- 오늘 낮: 10/29 주간보고 (w7 숫자 21% → 53%) 를 LAB-1309 `09_주간보고/2026-10-29/` 에 올림. 저장소 정리 (맨 위 문서 9개 → `10_문서/`, README 새로 씀, 일지 목록 달마다, 06_결과물/PiPER 맨 위 표). **LAB-1309 는 /home/user/ACT/LAB-1309 가 살아 있는 clone** (~/smolVLA/LAB-1309 는 10/21 에 멈춘 옛 clone)
- 다음: w8 결과 일지 → 일 2개가 안 오르는 이유 (넘긴 뒤 B 를 못 쥠? A 내려놓기?) 나눠 보기

## 10/29 00:05
- 사용자 "오늘은 여기까지" (10/29 00:05) → 매시 정기 점검 (cron) 끔. `piper-w8` 은 그대로 돎 (끼어들기 DAgger 설정마다 ~125/200 → w8 학습 → 평가, 아침 9시 반쯤 끝 예상). 다시 시작하면 outputs/piper/world8.log 끝의 표·SUMMARY 부터 보고
- 그때까지 정확도 (w7, 일 1개 150번): 혼자 3cm 32 (21%)·판 안 47, 다듬기 3cm 80 (53%)·판 안 83. 일 2개 (w7 + 다듬기, default 20): 둘 다 6/20, 일별 18/40 (w6r 2/20, 11/40)

## 10/28 22:00
- **w7 (w6 에서 15000, + DAgger 5차 433, val 10k 0.0815 → 15k 0.0839)**: 혼자 10·13·9 = 32/150 (21%), 다듬기 31 (62%, 17.3%) · 25 (50%, 11.7%) · 24 (48%, 23.2%) = **80/150 (53%)**. w6 와 거의 같음 (random 만 오름). 일 1개 DAgger 되풀이는 혼자 20% 근처에서 멈춤
- 일지 10/28 4·5절 push (5e94586)
- `piper-w8` 21:02 끼어들기 DAgger 시작 (w7) ‖ w7r 일 2개 default

## 10/28 12:10
- **w5 + SURE (평가만)**: 혼자 11·4·13 = 28/150, 다듬기 29·20·28 = **77/150** (w5r 70). many 다듬기 22 → 28
- **w6 (w5 에서 15000, wfeat3s = SURE 로 다시 계산한 학습 데이터 11,398 + DAgger 4차 423, val 0.0855)**: 혼자 11·9·10 = 30/150 (20%), 다듬기 32/50 (64%, 16.8%) · 20/50 (40%, 13.8%) · 27/50 (54%, 22.3%) = **79/150 (53%)**. 표 outputs/piper/world6.log 끝
- w6r 대 w6: 다듬기만 성공 56, 혼자만 성공 7 (1018·1001·1010·1047·1015·1023·1029). 그중 4개는 다듬기가 다른 물체를 쥐게 함 — scratchpad intent/diag.py (many 1015): VLA 가 손을 10~12cm 높이로 다른 물체 위를 지나가거나 헤맬 때 _target (손 아래 가장 가까운 물체, 윗면 + 10cm 안) 이 그 물체를 잡아 맞추기·쥐기를 함. 고칠 것: VLA 동작이 그 물체 쪽으로 가고 있을 때만 / 벌린 손일 때만
- w6 혼자 실패 (다시 나눔): 못 쥠 27, 쥐었다 1~19스텝 만에 놓침 23 (판 아닌 곳에 놓은 것 대부분이 이것 — 놓기가 아니라 쥐기가 약함), 판 안 3cm 밖 29, 판 근처 15cm 안 19, 들고 가서 먼 곳 13, 들고 끝 6. w6r: 못 쥠 28 이 가장 큼
- **검은 물체**: w6r 성공 3/22 (파랑 16/20), 쥠 10/22. 막대 14/38. default 1000 (검은 막대) 추적 (scratchpad intent/diag2.py, diag3.py, img/): VLA 가 막대 끝을 쥠 (손이 가운데에서 6cm) → 다듬기 '쥠' 판단 (오므린 손 4cm 안) 실패 → 놓기 다듬기 안 켜짐. 들어 올린 막대는 위 카메라에서 시차로 8cm 어긋남. 시도 (되돌림, 3장면 결과 그대로): ① 가린 덩어리도 fresh 에 ② 쥠 거리 = 물체 반 길이 + 1.75cm (outputs/piper/{world_assist_freshoccl_heldr_tried,grasp_assist_heldr_tried}.py). 다음: 손 위치를 eef 대신 손가락 끝 (FK) 으로 쥠 판단, 들어 올린 물체 시차 고치기 (높이 알면 pix2table 에 그 높이)
- **다듬기 HEADING 시험 (12:00~12:47, 실패 → 되돌림)**: _target 을 '손 활짝 벌림 + VLA 동작이 그 물체 쪽이거나 거의 멈춤', 정하면 곁에 있는 동안 고정 으로 좁힘. w6 many 27장면: 11/27 (전 16/27), 새로 실패 5 (1001·1009·1010·1013·1018), 새로 성공 0, 다른 물체 쥠 7 (전 6) → 끄고 원래대로 (시도 코드 outputs/piper/world_refine_heading_tried.py, 결과 outputs/world_w6rh_many)
- **일 2개 끼어들기 (w6 + 다듬기, default 1000~1019, outputs/world_w6r_default/k2n.json)**: 둘 다 성공 2/20, 판 안 5/20, 일별 11/40. A 를 30스텝 쥐어 B 로 넘어감 14/20, 넘어간 14장면에서 B 성공 5, A 성공 6. 넘어간 뒤 B 를 아예 못 쥔 장면 6 (B held 0). 1004 는 다듬기가 A 를 내려놓을 곳을 파란 정육면체(쌓기)로 잡음 → 빈 자리 내려놓기 확인 필요
- **끼어들기 DAgger piper_sim/world_dagger_nested.py (18시)**: 일 2개, VLA 가 A 를 30스텝 쥐면 B 지시 (못 쥐면 시범 프로그램이 쥐어 옮기는 도중까지, 기록 안 함) → VLA 가 B 지시로 0~300스텝 → 시범 프로그램이 A 를 들고 있으면 가까운 빈 자리에 내려놓기 → B → A 지시로 돌아가 A. 선생 구간만 지시문별 N{seed}_t{T}_s{i}.npz. 시험 3장면 2 성공
- `piper-w8` (run_world8.sh, log outputs/piper/world8.log): piper-w7 끝나면 끼어들기 DAgger (w7, 세 설정 × 200, 11000~11599, data/world_daggerN_{cfg}) ‖ w7r 일 2개 default 20 → wfeat2s·wfeat3s·사진 특징 → w8 학습 (w7 에서 15000) → w8 일 1개 혼자·다듬기 50, 일 2개 다듬기 20 (세 설정). 내일 아침
- 일지 10/28 push (6b8d805): 그림 outputs/media/world_w6_progress.png (scratchpad fig_w6.py), 4K outputs/media/world_w6r_s1020.{mp4,gif} (default 물체 8개, 개입 9/239)
- `piper-w7` (run_world7.sh, log outputs/piper/world7.log, 11:50): DAgger 5차 (w6, 세 설정 × 200, 10350~10949, data/world_dagger5_{cfg}) → 사진 특징 ‖ wfeat2s → wfeat3s → w7 학습 (w6 에서 15000) → 혼자·다듬기 50장면 → 표. 밤 9~10시 끝 예상

## 10/27 22:00 (가장 최신, 이것부터)
- **w5 (w4 에서 15000, wfeat3 가까이 보기, val 0.0944)**: 혼자 default 12/50 · random 5/50 · many 10/50 = 27/150 (18%, 13~25%; w4 20/150). 다듬기 29/50 (58%, 개입 18.4%) · 19/50 (38%, 14.2%) · 22/50 (44%, 26.8%) = **70/150 (47%)** (w4r 64/150). 차이는 우연 범위. 표 outputs/piper/world5.log 끝
- 놓기 분석 (w5, 판 20cm 안에 놓은 82개): 놓은 손이 판에서 로봇 쪽으로 평균 3.1cm (사분 −7~+1.7cm) — 한쪽 치우침보다 흩어짐. 판 기억 오차 0~0.7cm (입력은 맞음)
- **w5r 실패 나누기**: many 못 쥔 17장면 중 11장면은 손이 물체에 가장 가까울 때 옮길 물체 기억이 17~57cm 틀림 (다른 물체 쥠 11). 첫 사진에서는 VLA 길 찾기가 13/14 맞음 (scratchpad vladet/diag.py; 1028 만 분홍 막대 → 분홍 정육면체). → 팔이 가리면 world_detect 의 지시문 물체 fallback (짝을 못 얻으면 남은/쓴 덩어리 중 그 이름 점수 가장 좋은 것) 이 같은 색 다른 모양을 고르고, 손에서 10cm 밖이라 world_feat 기억을 덮어씀
- **고침 시험 중 (21:55)**: world_detect.detect 가 fallback 으로 채운 이름을 self.fallback 에 남김 (동작 같음), world_feat.SURE (환경 변수 PIPER_WF_SURE=1) 이면 fallback 찾기는 이미 있는 기억을 덮지 않음. 기본은 꺼짐 (학습 데이터와 같은 길). 백업 outputs/piper/{world_detect_before_fallbackflag,world_feat_before_sure}.py. `piper-sure`: w5 혼자 many 실패 10장면 (outputs/world_w5sure_many_t)
- `piper-d4` DAgger 4차: default 200 끝 (이어받아 성공 139, VLA 가 혼자 끝냄 21), random 끝 (149, 12), many 진행 중
- **SURE 시험 결과 (w5 혼자, many 실패 10장면)**: 손이 가장 가까울 때 옮길 물체 기억 오차 16~72cm → 9/10 이 0.1~2.5cm, 다른 물체 쥠 0. 쥔 장면 0 → 4, 성공 0 (어려운 장면들). → SURE 를 학습 데이터에도: WorldFeat(sure=), world_feat_cache --sure (*.wfeat2s / *.wfeat3s), train_lora --feat wfeat3s → worldfeat_sure 표시 → Runner 가 sure 로 (없으면 PIPER_WF_SURE). 백업 outputs/piper/{train_lora,switch_experiment,world_feat_cache}_before_sure.py. many 시범 49개 표본: 옮길 물체 기억이 3cm 넘게 달라지는 프레임 쥐기 전 13%, 쥔 뒤 38% (쥔 뒤에는 옛 길이 다른 물체를, SURE 는 집기 전 자리를 기억)
- `piper-w6` (run_world6.sh, log outputs/piper/world6.log, MemoryMax 24G): ① w5 + SURE 혼자 50장면 (outputs/world_w5s_{cfg}) → piper-d4 끝나면 다듬기 (world_w5sr_{cfg}) ‖ 학습 데이터 wfeat2s (CPU 3줄, nice) ② d4 끝나면 d4 wfeat2s ③ wfeat3s ④ w6 학습 (w5 에서 15000, wfeat3s, + d4) ⑤ w6 혼자·다듬기 50장면 ⑥ 표. 아침 8~9시쯤 끝 예상

## 10/27 17:50
- **빼보기 (w4, default 50장면, outputs/world_w4{rg,rp,a}_default)**: 혼자 8/50 (16%) · 다듬기 쥐기만 10/50 (20%, 11~33%, 개입 17.1%) · **놓기만 17/50 (34%, 22~48%, 개입 1.0%)** · 쥐기+놓기 27/50 (54%, 40~67%, 16.8%) · 예전 보조 (지시문 읽음) 24/50 (48%, 35~61%, **55.9%**). → 놓기 다듬기는 동작 1% 만 바꿔 두 배, 다듬기가 예전 보조보다 개입 1/3 로 같거나 높음
- w5 학습 14000스텝 (val 5k 0.0961, 10k 0.0973) → 18:30쯤 평가
- `piper-d4` (run_world_d4.sh, log outputs/piper/world_d4.log, MemoryMax 10G): w5 평가가 시작되면 DAgger 4차 1줄 (w5, --t 80 450, default 9750~9949 → random 9950~10149 → many 10150~10349, data/world_dagger4_{cfg}) → wfeat2 → wfeat3 → piper-w5 끝난 뒤 사진 특징. 다음 학습 (w6) 은 w5 성적 보고 정하기

## 10/27 14:00
- **w4 + 다듬기 50장면 (outputs/world_w4r_{cfg})**: default 27/50 (54%, 40~67%, 판 안 29, 개입 16.8%) · random 16/50 (32%, 21~46%, 18, 13.5%) · many 21/50 (42%, 29~56%, 22, 18.6%) = **64/150 (43%, 35~51%)**. 혼자 20/150. 실패: 쥐었지만 못 놓음 49 (그중 34개는 끝 자리가 판에서 20cm 넘음 — VLA 가 판 아닌 곳에 놓거나 들고 끝남, 대부분 first_dest None), 다가갔지만 못 쥠 24
- 그림 outputs/media/world_w4_refine.png (scratchpad fig_w4.py), 4K outputs/media/world_w4r_s1046.{mp4,gif} (default 빨간 막대, 개입 32/320). **일지 10/27 3·4절 push (caef0ed)**
- `piper-w5` 학습 13:44 시작 (학습 9878 / 검증 1097 구간, 3판 칸 [11:15] 통계 std 0.80~0.88). 17시쯤 끝 → 50장면 혼자·다듬기
- `piper-w4abl` (run_w4_ablation.sh, log outputs/piper/w4_ablation.log, 평가 1줄씩 — 학습과 GPU 나눔): w4 default 50장면 다듬기 쥐기만 (world_w4rg_default) → 놓기만 (world_w4rp_default) → 예전 보조 (world_w4a_default) → 표

## 10/27 12:20
- **w4 (w3 에서 15000, DAgger d3 583구간 더함, val 0.0950) 일 1개 단독 50장면 (1000~1049)**: default 8/50 (16%, 95% 8~29%, 판 안 9) · random 5/50 (10%, 4~21%, 판 안 9) · many 7/50 (14%, 7~26%, 판 안 13) = 20/150 (13%, 9~20%). 같은 20장면 (1000~1019) 은 w3 와 똑같음 (2·2·1)
- 실패 나누기 (world_fail): 쥐었지만 못 놓음 54·60·54%, 다가갔지만 못 쥠 16·20·18%. **지시한 물체를 쥠 42·40·40/50 (w3 random 10/20 → w4 random 17/20)** — 이제 쥐기는 되고 놓기가 문제. 쥐고 실패한 것의 끝 거리 중앙 8.1·11.9·20.5cm, 15cm 넘음 13·17·17 (다른 판에 놓았는지 아직 모름)
- **world_eval 기록 추가 (12:10, 백업 outputs/piper/world_eval_before_release.py)**: 일마다 end_pad·end_pad_cm (끝난 물체에서 가장 가까운 판), release (그 물체를 놓은 손 xy, 5개까지), release_pad (놓은 자리에서 가장 가까운 판·거리). 동작은 그대로. 시험 1003: 놓은 손은 판 6.3cm, 물체는 2.7cm (손목 기울여 쥐어 손-물체 xy 가 다름)
- `piper-w4rel` (12:15): w4 단독 many 1000~1019 다시 (새 기록으로, outputs/world_w4rel_many) — 놓기 실패가 판 고르기인지 마지막 몇 cm 인지
- **w4rel many 1000~1019 (새 기록)**: 1/20, 판 안 4. 쥐고 놓은 12장면 중 맞는 판 쪽으로 가서 판 가운데에서 3~14cm 에서 폄이 대부분. 놓을 때 VLA 의 놓을 판 위치 입력(기억) 오차 0.1·2.1cm (1001·1002) → 위치 입력은 맞고 VLA 가 마지막 몇 cm 를 못 맞춤. world_eval 에 feat_err_cm (obj: 쥐기 전 손이 가장 가까울 때 물체 기억 오차, dest: 놓을 때 판 기억 오차, dest_end), dest_xy·obj_xy 도 남김
- **위치 입력 3판 (12:40, 백업 outputs/piper/{world_feat,train_lora,switch_experiment,world_feat_cache}_before_v3.py)**: world_feat.derive_v3 — [11:15] = tanh((옮길 물체·놓을 곳 − 손) / 판 크기 절반 0.045). [0:4] 는 ±60cm 라 정규화하면 1cm ≈ 표준편차 0.05 → 가까이 보기 칸. 사진 없이 2판에서 바로 (world_feat_cache --version 3 --from-v2, 10975 파일 10초). train_lora --feat wfeat3 → worldfeat_v2·worldfeat_v3 표시 → Runner WorldFeat(version=3). w4 state_proj 의 [11:15] 열은 0 (시작 때 그대로), state_suffix_proj 는 0 아님 (작은 흔들림)
- `piper-w5` (run_world5.sh, log outputs/piper/world5.log, MemoryMax 24G): piper-w4 끝나면 w5 학습 (w4 에서 15000, wfeat3, 데이터 w4 와 같음 — 3판 효과만) → 일 1개 단독·다듬기 50장면 (outputs/world_w5_{cfg}, world_w5r_{cfg}) → 표 (w4, w4r, w5, w5r)
- `piper-w4` ⑥ 다듬기 50장면 진행 중 (outputs/world_w4r_{cfg}; 12:35 default 11/23, random 4/15, many 4/15) → world_table (w3, w3r2, w4, w4r)

## 10/27 02:30
- **w3 (w2 에서 15000, wfeat2, DAgger d2 621구간 더함, val 0.0894)** 일 1개 (1000~1019): 단독 default 2/20 · random 2/20 · many 1/20 (w2 1·0·0), 다듬기 7/20 · 2/20 · 3/20 (w2r 7·3·1). 막대 다듬기 성공 2/16 (w2r 0)
- **다듬기 버그 (10/27 02시 고침)**: 다듬기를 켜고 쥐었는데 실패한 장면 대부분이 first_dest None (놓기 다듬기가 안 켜짐). 원인: world_detect._rescue (보조 길 두 번째 찾기) 가 그리퍼 밑에 가려진 물체 대신 로봇 가리개 밖으로 나온 손목 카메라 몸체를 그 물체로 골라 보조 기억이 9cm 틀림 → '쥠' 판단(손 4cm 안) 실패 → 놓기 다듬기 안 켜짐. 고침: _rescue 가 로봇 가리개(3픽셀 넓힘)에 붙은 덩어리는 안 고름 (백업 outputs/piper/world_detect_before_rescuefix.py). 1013 실패 → 성공 (0.5cm). 정지 장면 시험 그대로. VLA 길은 _rescue 를 안 써서 학습 데이터 영향 없음. **10/26 의 다듬기·예전 보조 점수는 이 버그가 있는 채로 잰 것**
- **w3 + 고친 다듬기 (w3r2)**: 6/20 · 2/20 · 4/20 (개입 20.8 · 17.1 · 46.2%) — 합 12/60 그대로. 원인: 놓을 곳을 '들고 있을 때 손 아래 가장 가까운 판·쌓을 물체' 로 잡아, 옮기는 도중 다른 물체 위를 낮게 지나가면 그 물체에 쌓거나, 3cm 밖이면 손을 못 펴게 해 수백 스텝 들고 있음
- **다듬기 놓을 곳 고침 (03:50, 백업 outputs/piper/world_refine_before_lock.py)**: VLA 가 손을 펴려 할 때(동작 그리퍼 < 0)만 정함, 판 우선 (판 반대각선 + 5cm 안), 판이 없을 때만 손 바로 아래 2cm 안 쌓을 물체, 정하면 놓을 때까지 고정(lock), 손 펴기 늦추기는 30스텝까지 (hold_max). 실패 4장면 (default 1006·1010·1012·1016) 시험 3/4 성공, 4/4 판 안, 놓을 곳 모두 맞음
- **w3 + 새 다듬기 (w3r3, outputs/world_w3r3_{cfg})**: default 8/20 (판 안 10, 개입 11.1%) · random 3/20 (5, 14.8%) · many 10/20 (10, 25.6%) = 21/60 (예전 다듬기 12/60, 단독 5/60). random 은 지시한 물체를 쥠 10/20 뿐 — 다음에 볼 것
- 4K: outputs/media/world_w3r_many_s1006.{mp4,gif} (w3 + 새 다듬기, many 물체 13개, 개입 9/145). 일지 10/27 push (fe97f35), 그림 outputs/media/world_w3_refine_fix.png
- **보조 기억 가림 고침 (09:30, world_assist._update, 백업 outputs/piper/world_assist_before_occl.py)**: 손 가까이에서 팔에 일부 가린 덩어리는 가운데가 틀어져 기억이 1.7~4cm 틀림 (random 1011 쥐기 실패). 이 장면에서 그 물체가 가장 크게 보인 픽셀 수의 85% 보다 작고 손 10cm 안이면 fresh·기억에 안 씀. 1011 쥠 (0 → 74스텝), default 5장면 3/5 (전과 같음). 예전 보조·다듬기 둘 다 적용 → **w4 평가부터 이 판**
- w4 학습 07:33 시작 (w3 에서 15000, DAgger d3 583구간 더함) → 50장면 평가
- `piper-w4` (run_world4.sh, log outputs/piper/world4.log): piper-w3 (일 2개 단독 평가 중) 끝나면 w3 + 고친 다듬기 일 1개 다시 (outputs/world_w3r2_{cfg}) → DAgger d3 (w3, 9000~9749) → wfeat2·사진 특징 → w4 학습 (w3 에서 15000) → 일 1개 단독·다듬기 50장면 (1000~1049)

## 10/26 15:00
- **w2 일 1개 (장면 1000~1019), piper_sim/world_table.py 로 표**: 기본 단독 1/20 · 예전 보조 6/20 · **다듬기 7/20 (개입 9.9%)** · 다듬기 쥐기만 2/20 · 놓기만 3/20 / 흔든 환경 단독 0 · 예전 보조 2/20 (개입 61.5%) · 다듬기 3/20 (5.3%) / 물체 많음 단독 0 · 예전 보조 2/20 (66.4%) · 다듬기 1/20 (13.9%). 지시한 물체를 쥠: 다듬기 16·12·12/20, 예전 보조 10·10·5/20. 막대는 다 0
- 일지 10/26 push (c624f34): 그림 outputs/media/world_w2_refine.png, 4K outputs/media/world_w2r_s1010.{mp4,gif} (다듬기 9/352 스텝)
- **위치 입력 2판 wfeat2 (16시)**: 막대는 방향을 맞춰야 쥠 (9cm > 벌린 손가락 7cm; w2 는 막대 6장면 모두 실패, 손은 0.1~1.4cm 위까지 감). world_detect 가 덩어리 긴 축 (_blob_axis → locate 가 self.yaw[이름] = (세계 방향 도, 길쭉함); 처음 보는 30장면 막대 중앙 0.6° 최대 1.5°, 정육면체 최대 3.2°, 각기둥은 못 잼 — 맞출 필요 없음). world_feat.WorldFeat(version=2): [8:10] sin·cos(kθ) (k = 360/world_assist.yaw_period: 막대 180°, 정육면체 90°), [10] 방향 있음. world_feat_cache --version 2 → *.wfeat2.npy. train_lora --feat wfeat2 → 모델 폴더 worldfeat_v2 표시 → switch_experiment Runner 가 WorldFeat(version=2). --renorm-new-feat: 늘 0 이던 칸(std<0.002)만 통계 다시 (가중치 그대로). 백업 outputs/piper/{world_feat_before_v2,train_lora_before_wfv2,switch_experiment_before_wfv2,world_detect_before_axis}.py
- `piper-w3` 다시 (16:00, run_world3b.sh, log outputs/piper/world3.log): DAgger d2 이어서 ‖ 기존 데이터 wfeat2 → d2 wfeat2·사진 특징 → w3 학습 (w2 에서 15000, wfeat2) → 일 1개 단독·다듬기, 일 2개 단독 (세 설정) → 표
- (예전) `piper-w3` 14:35 DAgger d2 시작 (logs outputs/piper/world_dagger2_{default,random,many}.log) → w3 학습 → 평가 (오늘 밤)
- 다음: 장면 50개 이상으로 다듬기 대 예전 보조·빼보기 다시, 막대 원인, k≥2 에서 다듬기가 빈 자리 내려놓기를 판으로 끌지 않는지

## 10/26 08:50
- **w2 학습 끝 (08:12)**: outputs/piper_w2_model/merged, val 0.1003 (5k) → 0.0933 → 0.0892 → 0.0917 → 0.0866 → 0.0853 (30k). 데이터 학습 8794 / 검증 977 구간 (default_t + random + many + DAgger d1 290 + r1 274)
- piper-world2p 가 평가 4줄에서 08:17 **OOM (단위 MemoryMax 10G)** → 평가만 `piper-w2eval` (run_world2_eval.sh, MemoryMax 22G, log outputs/piper/world2_eval.log) 로 다시. 세 설정 × k=1~4 × (단독, + 보조) → outputs/world_w2_{cfg}, outputs/world_w2a_{cfg}, 성적표 outputs/piper/world_board_w2_{alone,assist}_{cfg}.md

- **w2 단독 default (10:00)**: k=1 1/20 (판 안 3/20), k=2~4 0/20 (일별 1/40, 3/60, 2/80). 실패 나누기 200개: 쥐었지만 못 놓음 66, 다가갔지만 못 쥠 38, 다가가지도 못함 7 (w1b 는 다가가지도 못함이 대부분 → 이제 놓기가 문제). k=1: 20/20 손이 3.5cm 안, 15/20 쥠, 쥔 것의 끝 거리 0.5~17cm
- **w2 + 예전 보조 default**: k=1 6/20 (30%, 판 안 6), k=2 1/11·k=3 0/6·k=4 0/10 (멈춤). 단독 random k=1~4 0 (판 안 2/20), many k=1~4 0 (판 안 1/20). 성공 6장면은 원기둥·삼각기둥, 막대는 단독·보조 모두 못 쥠
- **논문용 다듬기 piper_sim/world_refine.py (10/26)**: WorldAssist 를 상속, _target/_dest 를 지시문 대신 'VLA 손 아래 가장 가까운 물체 (윗면 + 10cm 안으로 내려왔을 때)' / '들고 있을 때 손 아래 가장 가까운 판·쌓을 물체 (판 반대각선 + 5cm 안, 윗면 + 든 물체 높이 + 10cm 안)' 로. 다가가기·엉뚱한 물체 막기·내려놓기·데려다 놓기·쌓은 것 피하기는 끔. world_eval --refine grasp,place | grasp | place. 결과 json 에 assist_steps·assist_changed (개입률, WorldAssist 도 셈)·first_intent·first_dest. 시험 3장면: intent 3/3 맞음, 개입 1~18%, 성공 0/3 (예전 보조도 이 3장면 실패)
- `piper-w2eval` 다시 (13:10, run_world2_eval_b.sh, log outputs/piper/world2_eval_b.log): 다듬기 k=1 세 설정 (outputs/world_w2r_{cfg}) + 예전 보조 k=1 random·many + 빼보기 쥐기만·놓기만 (world_w2rg_default, world_w2rp_default)
- `piper-w3` (run_world3.sh, log outputs/piper/world3.log, MemoryMax 22G): piper-w2eval 끝나면 DAgger d2 (w2, --t 40 450, 세 설정 × 250장면 8000~8749 → data/world_dagger2_{cfg}) → wfeat·사진 특징 → w3 학습 (w2 에서 15000) → k=1·2 단독 + k=1 보조 (outputs/world_w3_{cfg}, world_w3a_{cfg})

## 10/25 20:30
- **w1b (w1 + 고친 wfeat 6000스텝, --proj-lr 3e-4 --new-lr 3e-4)**: 일 1개 단독 1/20, + 보조 2/20 (판 안 3/20). val 0.1059. 실패 나누기 (piper_sim/world_fail.py, world_eval 이 일마다 held_steps·hand_min_cm·end_dist_cm·held_other 남김): 단독 다가가지도 못함 12, 다가갔지만 못 쥠 3, 쥐었지만 못 놓음 4. 위치 입력 바꿔 넣기 시험 (scratchpad probe_wfeat.py): 방향은 따름 → 정확도 문제 (5~13cm 앞에서 손 닫음). 보조는 물체 13cm 위까지 데려다 놓고 돌려주는데 VLA 가 안 내려감
- **DAgger** piper_sim/world_dagger.py (WE.install() 꼭 — 안 부르면 시범 프로그램 쥐기가 다 실패): w1b 20~300스텝 → 시범 프로그램이 이어받음, 선생 구간만 D{seed}_t{T}_s0.npz. 시험 6/6. `piper-dagger` (run_world_dagger.sh): default 7000~7499 → data/world_dagger_d1, random 7500~7999 → data/world_dagger_r1 (log outputs/piper/world_dagger_{d0,r0}.log)
- `piper-vis1`: 사진 특징 shard 1 (w1b 때 GPU 부족으로 멈췄던 것), `piper-world2p`: shard 0 → 끝나면 run_world2_train.sh: piper-vis1·w1b 기다림 → 사진 특징 남은 것 → **piper-dagger stop** → DAgger wfeat·사진 특징 → w2 학습 (w1b 에서 30000, 데이터 default_t+random+many+DAgger) → 세 설정 × k=1~4 × (단독, + 보조) → 성적표 outputs/piper/world_board_w2_*
- 22:50 사진 특징 shard 1 이 느려 `piper-vis1r` (vis_cache --reverse, 뒤에서부터) 추가, `dagger-stopper`: piper-vis1 이 끝나면 piper-dagger stop (run_world2_train.sh 의 사진 특징 2줄과 GPU 메모리 겹치지 않게). vis_cache 임시 파일 이름에 pid
- 4K: outputs/media/world_w1ba_s1008.{mp4,gif} (w1b + 보조 성공, GIF 32.3MB) → 일지 10/25 push (1d9d4a1). w2 학습 01:37 시작 (학습 8794 / 검증 977 구간, 0.8초/스텝 → 08:20쯤 끝)
- 일지 10/25 1~4절 push (dfa1c37)
- **하드코딩 걷어내기 (22시, 동작은 그대로 — 읽는 곳만)**: ① world_feat.parse_instruction 이 설정 instruction 틀에서 정규식을 만듦 (백업 outputs/piper/world_feat_before_template.py; 지시문 13,137개 예전과 같음) ② grasp_assist 손목 한계 LIMIT_DEG 를 로봇 XML joint4~6 에서, 손가락 0.028/0.005/0.03 을 그리퍼 XML finger_joint1 범위의 0.8 / 1/7 / 6/7 로 (Q_HELD/Q_EMPTY/Q_WIDE, 백업 outputs/piper/grasp_assist_before_model.py; 값 같음)
- **남은 하드코딩 (w2 평가 뒤에 — 동작이 바뀜)**: 책상 높이 TABLE_Z 0.90 (world·grasp_assist, scripted_expert 0.898) → 시뮬 모델에서 재기 / 보조 높이들이 블록 4.5cm 기준 (다가가기 13cm·옮기기 10cm·정렬 반경 4cm 높이 10cm 등) → 물체 높이 기준 / 그리퍼 길이 0.12 (world.reach_map)·로봇 가리기 손끝 0.13 (world_detect.RobotMask) → 그리퍼 XML eef 0.12·finger 0.135 / switch_experiment GRIPPER_OPEN_QPOS 0.035 / grasp_assist 가 안 쓰는 front_calib.json 을 읽음

## 10/25 15:00
- **w1 단독 0%**: k=1~4 모두 0/20 (판 안도 0), 모든 장면 600스텝 끝까지 아무것도 못 듦 (outputs/world_w1, 성적표 outputs/piper/world_board_w1.md). 진단 (scratchpad diag_w1.py, 장면 1017): 평가 wfeat 는 정확 (참값과 0.5cm 안), VLA 가 엉뚱한 물체·허공으로 감. 원인 가설: w1 학습 wfeat(예전 찾기)는 지시문 물체를 '아예 못 찾음'이 원기둥 31.6%·정육면체 23%·삼각 22%·육각 15% (모양을 잘못 짝지어), '지금 보임' 11~42% → VLA 가 위치 입력을 쓰지 않게 배움. 새 wfeat 는 아예 못 찾음 ≤4%
- piper-worldc 는 w1 + 보조 평가 시작 직후 stop (14:46, 쓸모없는 GPU 시간). w1 + 보조는 안 잼
- `piper-wfeat2` 다시 (14:50): random 2 + many 2 + default 4줄 동시 (nice 10), 예전 default wfeat 는 data/world_default_t_oldwfeat
- w1 state_proj 열 크기: 로봇 상태 0:8 ≈ 3.35, 물체 입력 8:16 ≈ 0.08~0.35 (0 에서 3e-5 로 자람) → train_lora `--proj-lr` 새로 (state_proj 학습률; 백업 outputs/piper/train_lora_before_projlr.py), `--new-lr` 은 state_suffix_proj
- `piper-w1b` (run_world_w1b.sh, log outputs/piper/world_w1b.log): default wfeat 다 되면 w1 에서 6000스텝 (고친 wfeat + --proj-lr 3e-4 --new-lr 3e-4) → k=1 단독 (outputs/world_w1b) ‖ k=1 + 보조 (outputs/world_w1ba). 이게 오르면 w2 (run_world2_train.sh, world2p 가 사진 특징 뒤 부름) 그대로, 안 오르면 w2 계획 다시
- `piper-world2p`: 14:48 부터 random·many 사진 특징 2줄 (GPU)
- 새 도구: piper_sim/world_vla4k.py (VLA 장면 4K 영상, world_eval 과 같은 장면·인자)

## 10/25 12:40
- **돌고 있는 것**
  - `piper-worldc` (run_world_round_c.sh, log outputs/piper/world_round_c.log): w1 학습 11000/20000 (val 0.1208 → 0.1191), 15시쯤 끝 → w1 단독 k=1~4 → w1 + 보조 → 성적표 (outputs/piper/world_board_w1*.md). **평가는 새 물체 찾기(아래)로 돈다 — w1 학습 wfeat 는 예전 찾기**
  - `piper-wfeat2` (run_world2_wfeat.sh, log outputs/piper/world2_wfeat.log): 새 찾기로 wfeat 다시, --stride 6 (학습이 읽는 장만; 다 계산한 것과 차이 최대 0.5cm). random·many 4줄 (~15:30) → w1 학습 끝나면 default_t (예전 것 data/world_default_t_oldwfeat)
  - `piper-world2p` (run_world2_prep.sh, log outputs/piper/world2_prep.log): piper-worldc 끝나면 random·many 사진 특징 2줄 (6장 간격) → run_world2_train.sh 가 있으면 이어서 (아직 안 만듦 — w1 결과 보고 만들 것)
  - 예전 wfeat: data/world_random_oldwfeat, data/world_many_oldwfeat (쓰지 않음)
- **물체 찾기 고침 (world_detect.py, 백업 outputs/piper/world_detect_before_tablefit.py · detect_*_before_tablefit.json)**: ① 사진마다 책상 자리 밝기로 맞춤 (ref_Y, 자동 노출처럼; 'classes' 는 맞춘 색, 'classes_raw' 는 손목 사진용) ② 이 사진의 책상 색 = 자리의 2차 함수 (가운데 60%로 맞춤) ③ 로봇 팔은 관절 각도로 가리므로 로봇·바닥 색 이름은 안 고름, 책상 밖은 버림 ④ 장면을 알면(보조) 그 장면의 색만 ⑤ 모양 특징에 윤곽 푸리에 1~8 + 공분산 가우스 (원기둥↔육각기둥 79 → 97%) + 배운 모양 밖(min_ll) 덩어리 버림 ⑥ VLA 길(장면 모름)은 그 색 모든 모양으로 짝지은 뒤 지시문 물체 (앞: 지시문 모양 하나만 → 같은 색 바뀜, 또는 모든 모양 → 지시문 물체를 다른 모양으로 읽어 못 찾음). ⑦ 보조 길(장면 앎)만 두 번째 찾기 _rescue: 못 찾은 것마다 안 쓴 덩어리 중 모양 맞고 색 거리(가운데 마할라노비스) 20 아래 (남색 판→파란 물체 색, 센 조명 주황 판→노랑). fit 150장면, 관측 사진 그대로. 처음 보는 30장면 (보조 길 못 찾음 / 3cm 넘음, VLA 길 못 찾음 / 3cm 넘음): default 3/0, 4/0 (255개) · many 6/1, 3/4 (435) · random 5/3, 15/2 (255; 앞 24/170 못 찾음). 그림 outputs/media/world_detect_random_fix.png
- 손목 사진: world_assist.wrist_det = Detector(space="raw"), classify(own_table=False)

## 10/25 04:30
- 사진 특징이 시간당 380구간 (GPU 100%, 512 사진 장당 ~70ms) → 남은 것은 6장 간격 (vis_cache.py: 저장 간격의 배수 허용, crc%6≡crc%3 이라 6장 간격 프레임은 3장 간격 안에 있음). piper-worldb 멈추고 **piper-worldc (run_world_round_c.sh, log outputs/piper/world_round_c.log)**: 사진 특징 6장 간격 → w1 학습 --frame-stride 6 → 평가

## 10/25 01:30
- 기본 설정 수집 끝 (00:49): 1,183장면 중 1,166 성공 (98.6%; 1s 272/272, 2n 206/207, 2s 100/100, 3n 200/204, 3s 94/97, 4n 204/211, 4s 90/92), 4,203구간·83만 프레임 → data/world_default_t
- wfeat 계산이 프레임당 60ms (10시간) → 지시문 두 물체 색만 찾기(world_detect.detect only=) + 로봇 가리기 cv2.circle → 21ms. piper-world·piper-world-after 를 멈추고 **piper-worldb (run_world_round_b.sh, log outputs/piper/world_round_b.log)** 로 이어감: wfeat 6줄 → 사진 특징 → w1 학습 → w1 단독 k=1~4 → w1 + 보조 k=1~4 → 성적표
- piper-world2: random 끝 (592/600, data/world_random), many 수집 중 (data/world_many, 일 1~6)
- 일지 10/24 9·10절 push (c46c7fa): np5v 효과 없음, 새 장면 손보기, 4K GIF (world_expert_k3)

## 10/24 16:40
- **사진 512 (설정 obs_size, world.install 이 LiberoEnv 기본값을 바꿈)**: 위 카메라 256 에서 물체 9픽셀 → 같은 색 원기둥·육각기둥 구분 불가. 512 에서 물체 찾기 (처음 보는 10장면): default 평균 0.25cm·95% 0.75cm, many(색 중복·6~14개) 95% 1.20cm (바뀜 1번), random 95% 1.23cm (못 찾음 10/88 — 색 흔들기·조명)
- 카메라 보정값: world.CAMERA_CALIB (randomize 가 카메라를 흔들면 그 값), world_detect 는 world.camera_pose() 를 씀. 시범 npz 의 scene.env_info.camera 에 저장 (world_collect), world_feat_cache 가 읽음
- 시범에 관절 각도 저장 (collect_expert.Rec "joints") → wfeat 계산 때 로봇 팔 가리기
- **시범 프로그램 고침 (16:05)**: 쥐는 높이 = min(max(h/2, 2.8cm), h−1.2cm) (설정 grasp.z_min·top_margin; 기울인 손이 2.25cm 에서 책상을 파고들어 후보가 다 빠졌다), ik.ArmIK.solve 40번마다 안 줄면 그만 (백업 outputs/piper/ik_before_stall.py; 쥐는 자세 찾기 70초 중 69초), world.chain 은 옮길 물체를 먼저 정하고 그것은 쌓기 받침으로 안 씀 (쌓은 받침을 나중에 옮겨 떨어짐). 고친 뒤 17/17 성공 (일 4개 겹침 9/9), 시간당 ~100장면
- 수집 다시 시작: piper-world (16:05), data/world_default (512, 관절 포함), 앞 256 데이터는 data/world_default_256old (쓰지 않음). 로그 outputs/piper/world_collect_{0,1,2}.log (앞 실행 world_collect_a_*)
- 팔 닿는 범위 지도: 열쇠 72f30ea5 는 지금 예전 지도 복사본 → planner-reach2 가 새 규칙으로 다시 재는 중 (outputs/piper/planner_reach2.log)
- piper-np5: np5v 혼자 10 끝 (17:39) → stop (쌍 평가 미룸). **np5v 80/200 · 판 안 118 = np5f 81 · 120 과 같음 (짝 +17 −18)**, np3 69 · 106. VLM LoRA 효과 없음. piper-world2 (random·many 수집) 17:47 시작
- 4K 시연 outputs/media/world_expert_k3.{mp4,gif} (시범 프로그램, 물체 8·판 3·일 3개 겹침, 성공, GIF 42.9MB 35장 colors 48 every 12) — piper_sim/world_demo4k.py
- world_board.py (성적표: k·방식·모양·물체 수), np5f + 보조 새 장면 시험 0/6 (새 장면 안 배운 모델이라 의미 없음, w1 로 다시)
- 16:20 고침: 각기둥 쥐는 방향 yaw = 면 법선 − 90 (앞: 육각기둥을 꼭짓점끼리 쥠 — 지금 도는 수집은 앞 코드), 손목 맞추기는 가장 긴 폭이 그리퍼(7cm) − 1cm 보다 작으면 끔 (육각·삼각·원기둥; 위에서 비스듬히 본 각기둥 윤곽으로 각도를 25° 틀리게 읽었다), 정육면체·막대 손목 각도 읽기는 방향 맞음
- planner_key 는 grasp 의 높이·폭 값만 (설명 글이 바뀌어도 같은 열쇠) → 지금 열쇠 6e388463 = 72f30ea5 복사본. **planner-reach2 가 끝나면 (72f30ea5 로 씀) 그 결과를 key 만 6e388463 으로 바꿔 복사할 것**
- `piper-world2` (run_world2_collect.sh): piper-np5 끝나면 random 2줄 (5000~5599) + many 1줄 (6000~6599, 일 1~6) → data/world_random, data/world_many (w2 재료)

## 10/24 15:30
- 일지 10/24 6~8절 push (2f85496): VLA 단독 점수, bf16 반올림 (np5f), 여러 모양·N개 장면, 손목 카메라 몸체
- **돌고 있는 것**
  - `piper-np5`: np5v 학습 (16:30 쯤 끝) → np5v 단독 혼자 10. 끝나 '[돌아오기 6쌍 → 나머지 6쌍]' 이 찍히면 짝 비교 보고 후 `systemctl --user stop piper-np5` (쌍 평가는 미룸, 자물쇠 확인)
  - `piper-world` (run_world_round.sh, log outputs/piper/world_round.log): 시범 수집 3줄 15:15~ (장면 3000~4199, 줄마다 장면당 ~110초 → 새벽 3시쯤), 그다음 다듬기·wfeat·(piper-np5 끝나면) 사진 특징·w1 학습 20000·w1 단독 평가 k=1~4
  - `detect-fit`: 위 카메라 물체 찾기 다시 배우기 (many → default → random, log outputs/piper/detect_fit.log)
- 팔 닿는 범위 (길 찾기, world_expert.planner_reach): −135~135° 모두 0.225~0.45m → piper_sim/worlds/reach_planner_73732d76.json (열쇠 = 로봇 파일·workspace·grasp 내용, 설정 이름과 무관)
- 시범 고침: world_expert.grasp 는 놓을 자리 조건으로 못 찾으면 조건 없이 다시 (다시 못 쥠 대책), 수집 초반 11/14 성공 (k=4 겹침 1/3)
- 새로 만든 것: `world_cmd.py` (한국어 해석기, 설정 낱말, 모양만 말해도 장면에 하나면 알아들음), `piper_sim/world_eval.py` (VLA 평가, 끼어들기 깊이 k, --switch-hold 30, 보조 옵션 그대로 받음 → world_assist), `piper_sim/world_assist.py` (GraspAssist 판단 그대로, 키 = 인스턴스 이름, 위 카메라 위치, 모양 대칭으로 손목 맞추기, 빈 자리 = 닿는 범위 지도 안), `piper_sim/world_feat.py` (VLA 입력 24칸 = 지시문의 옮길 물체·놓을 곳 − 손; 모델 폴더 worldfeat 표시 → switch_experiment.Runner 가 씀), `world_feat_cache.py`
- 물체 찾기 (world_detect): 인스턴스 단위 (같은 색 다른 모양은 덩어리 모양 특징으로 짝짓기), 로봇 팔 자리 가리기 (관절 각도 → 정기구학 → 투영, RobotMask), 색 판정 표(64³). many.json (색 중복, 물체 6~14) 첫 시험: 같은 색 원기둥↔육각기둥이 바뀜 (256 사진에서 9픽셀), 회색 판↔로봇 팔
- 설정: default.json (색 겹치지 않음, 물체 2~8), random.json (조명·색·책상·카메라 흔들기), many.json (색 중복, 물체 6~14, 판 3~6)
- switch_experiment.py 고침 (백업 outputs/piper/switch_experiment_before_worldfeat.py): Runner.wf (worldfeat 표시 모델만), Episode 시작 때 wf.reset. train_lora.py: wfeat 이면 worldfeat 표시

## 10/24 15:00
- **np5f (np3 + --fp32-master) 단독 혼자 10: 3cm 34.5 → 40.5% (81/200), 판 안 53 → 60%. 짝 +45 −33 (p 0.21), 판 안 +39 −25 (p 0.10)** — 블록 위에서 못 쥠 29 → 19, 놓기 실패 79 → 80 그대로 (위치를 흐릿하게 앎 → np5v 가설)
- 돌고 있는 것: `piper-np5` 는 np5v 학습 중 (14:1x 시작, vram 6.9GB, 1000스텝 ≈ 13분) → np5v 단독 혼자 10 → 쌍들
- **사용자 (13~15시) 새 방향: "객체도 여러모양으로 n개씩 … 무제한", "piper 360도", "하드코딩 아닌걸로", "실제 사물 접시도 쉽게?", "모든 변수 환경 95% 이상", "손목에 카메라도 보이게 모델링"** (memory world-goal)
- 새 파일 (예전 블록 장면 파일은 안 건드림, np5 가 씀):
  - `piper_sim/worlds/default.json` 설정 (모양 5: cube·bar·cylinder·hexprism·triprism, 물체 색 8, 판 색 6, 개수 범위, 카메라 위(0,0,1.75)·fovy 75, 책상 1.0×1.2, 쥐는 높이 규칙, 손목 카메라 D405 크기)
  - `piper_sim/world.py`: 자산·LIBERO 등록, 팔 닿는 범위 reach_map (robot.xml 관절 한계 + 역기구학, 1번 관절 ±147° → 각도 −135~135°, 반지름 0.15~0.45), sample_scene(seed), tasks_for, chain, Suite(장면마다 BDDL), place, 로봇 책상 가운데, 카메라 설정대로, make_cam_robot (assets/robots/piper_noped_cam_default)
  - `piper_sim/world_expert.py`: 모양별 쥐는 방향, run_chain(nested 깊이 제한 없음, say=지시 바뀜), why(실패 원인)
  - `piper_sim/world_collect.py`: 지시문별 구간 저장 (W{seed}_k{k}{n|s}_s{j}.npz), 시험 3장면 모두 성공
  - `piper_sim/world_detect.py`: 분할 그림으로 색 분포 배우기 → piper_sim/worlds/detect_default.json, 넓이로 로봇 팔 덩어리 거름. 처음 보는 20장면 170개 중 165 찾음, 오차 중앙 0.31cm, 95% 1.1cm
  - `piper_sim/world_view.py`, `world_chain_test.py` (outputs/media/world_view.png, wrist_cam_model.png)
- 시범 시험 (world-chain1, log outputs/piper/world_chain1.log): k=1 15/15, k=2 겹침 6/10 (다시 못 쥠 3, 삼각기둥 못 쥠 1) → world-chain2 (2020~2039, 원인 기록) 돌고 있음
- 다음: 시범 실패 고치기 → 대량 수집 → VLA 학습(world) → 보조·한국어 해석기 설정 기반으로 → 조건별 95%

## 10/24 10:40
- **사용자 (10시): "보조 없이 VLA 정확도 올리는 방법 찾아줘"** — VLA 단독 새 기준 np3 28항목 23.2% (혼자 34.5)
- np3 단독 혼자 과제 실패 131 = 놓기 79 (판 안 3cm 밖 37 + 판 밖 42), 블록 위에서 못 쥠 29, 쥐었다 빠짐 15. 놓는 순간 손-판 가운데 3.5~4.4cm, 손은 멈춤, 어긋남 방향은 흩어짐 (평균 1cm) = 위치를 흐릿하게 앎
- 얼린 정면 카메라 토큰(cache/vis) → 물체 위치 선형 탐침: 같은 분포 1.9cm, 넓은 장면 3.5cm (scratchpad probe.py; 비선형 탐침 probe2 는 CPU 가 느려 멈춤)
- **발견: 동작 전문가 가중치 257개가 bfloat16 (from_pretrained 가 그렇게 올림) → 3e-5 갱신이 반올림에 지워짐.** np2→np3 10000스텝에 바뀐 값 32.4%, |w|≥0.03 은 0%. EMA 사본도 bfloat16 (memory bf16-rounding)
- 새 옵션 train_lora.py (백업 outputs/piper/train_lora_before_vlmlora.py): `--fp32-master` (float32 원본에 갱신·EMA), `--vlm-lora 16 --vlm-lora-lr 2e-4` (VLM 32층 LoRA, 저장 때 합침). CPU 4스텝 시험 통과
- **돌고 있는 것: `piper-np5` (run_np5v_round.sh, log outputs/piper/np5_round.log)**: piper-alone 끝 기다림 → np5f 학습 (np3 + --fp32-master, 같은 데이터 10000) → np5f 단독 혼자 10 (piper_np5fh) → np5v 학습 (+ --vlm-lora 16) → np5v 단독 혼자 10 (piper_np5vh) → 둘 다 돌아오기 6쌍·나머지 6쌍 → 성적표
- 비교: `python cmp_rounds.py outputs/piper_np3mch outputs/piper_np5fh --names np3 np5f`, `python fail_classify.py outputs/piper_np5fh`

## 10/24 05:30
- **목표 달성 (사용자 "셋 다 성공도 90 넘기자"): np3 + 고친 보조 전부 셋 다 240장면 217 = 90.4%** (1000~1019: 107/120, 처음 보는 1020~1039: 110/120; 전환X 4 빼면 91.9%)
- 같은 설정 28항목 (piper_np3d, 05:00 끝): **94.5% / 판 안 94.6%** (혼자 98.0, 일 바꾸기 92.9, 돌아오기 91.7, 셋 다 89.2). outputs/piper/score_piper_np3d*.md
- 설정: korean_live.py 의 POLICY (np3) + ASSIST 목록이 그대로 이 설정 (+ PIPER_FRONT_CALIB 넓은 변환식, 평가 --release-wait 40)
- 4K 시연 3장면 모두 성공 (outputs/media/k4_all3_{1019,1005,1010}), 일지 10/24 (push 0c2a492) 에 1019. render4k 카메라 elevation -38, distance 0.96, lookat (-0.05,0,1.0)
- 돌고 있는 것 없음. 다음 후보: 미뤄 둔 VLA 단독 평가 (outputs/piper/deferred_alone.txt), 전환X (A 를 먼저 끝냄) 를 없애는 평가 방식, 실물 PiPER 실행기

## 10/24 02시
- **np4 (새 기준 보조 시범 461개로 np3 에서 더 학습, outputs/piper_np4_model) 는 셋 다 83.3% (100/120) — np3 86.7% 보다 낫지 않음** (돌아오기 실패 3 → 9). 돌아오기 6쌍 + 몇 항목만 재고 멈춤 (piper_np4x)
- 가장 좋은 설정의 돌아오기 실패 5 중 2 (A8→B9 1014, A1→B7 1019): 빨강을 든 채 목표 판 6~15cm 에서 500스텝 맴돎 → ㉓ `--deliver` (든 채 4초 동안 놓을 곳에 1cm 도 안 다가가면 ⑫ 와 같은 동작으로 놓을 곳 위로 옮겨 놓음; 쌓은 블록 있으면 14cm 높이로)
- 돌고 있는 것: `piper-np3d` (run_np3d_round.sh, log outputs/piper/np3d_round.log): np3 + 새 설정 + ㉒' + ㉓, 돌아오기 6쌍 (outputs/piper_np3d_switch). 비교 = piper_np3x (쌓기 두 쌍은 piper_np3h) 셋 다 105/120
- 나머지 실패: 일 바꾸기 전에 A 를 끝냄 (전환X) 3, 블록 위에서 못 쥠, 3cm 살짝 넘게 놓음 (3.2·3.5cm)
- **02:39 np3 + 새 설정 + ㉒' + ㉓ (piper_np3d): 셋 다 107/120 = 89.2%** (전환X 3 빼면 107/117 = 91.5%). B 실패 4, 돌아오기 4, B 무너짐 2, 전환X 3. A1→B7 20/20. 짝 비교 np3x 대비 4:1
- **이 설정 = np3 + BEST + ⑲ --stuck-grasp 40 + ⑨' --yaw-none-reset 10 + ⑰' --mem-tol 0.012 + ⑳ --open-empty + ㉒' --stack-guard --stack-clear --guard-r 0.07 + ㉓ --deliver** (+ 넓은 변환식, --release-wait 40)
- **03:30 처음 보는 장면 1020~1039 (piper_np3d2): 셋 다 110/120 (91.7%). 합계 240장면 217 = 90.4% (95% 구간 86~93.5), 전환X 4 빼면 91.9%. B 94.6%, 돌아오기 93.8% → 목표 달성 (사용자에게 알림)**
- 돌고 있는 것: `piper-np3d3` (run_np3d3_round.sh): 같은 설정 28항목 나머지 (6쌍 + 혼자 10) → outputs/piper_np3d, 끝나면 성적표
- 그다음 `piper-k4c` (run_k4c.sh): korean_live (POLICY np3, 보조 목록 = 이 설정, render4k 카메라 elevation -38 distance 0.96 lookat (-0.05,0,1.0)) 로 노란 막대 끼어들기 4K, 장면 1019 → 실패면 1005·1010 (outputs/media/k4_all3_*, log outputs/piper/k4c.log)
- 끝나면: 성적표 (score_piper_np3d) + 일지 10/24 + 4K GIF push

## 10/23 18:30
- np3 + 새 설정(⑲⑨'⑰'⑳) 28항목 (piper_np3x, 17:45 끝): **93.4%** (혼자 97.5, 일 바꾸기 90.4, 돌아오기 92.5, 셋 다 86.7). np2 + 고친 보조 93.6% 와 같은 수준
- 쌓기 두 쌍 ㉒' `--stack-guard --stack-clear --guard-r 0.07` (piper_np3h): 셋 다 31 → 32/40, B 무너짐 5 → 2. 6쌍으로 치면 약 105/120 = 87.5% — 보조를 하나씩 고쳐도 ±1~3장면
- **돌고 있는 것: `piper-np4` (run_np4_round.sh, log outputs/piper/np4_round.log)**: ① np3 + 새 설정 + ㉒' + ㉑ 로 새 기준 보조 시범 (장면 3320~3339 × 22, data/piper_np_assist2) ② 정리 ③ np4 학습 (np3 에서, 보조 시범 1·2, 10000스텝, outputs/piper_np4_model) ④ np4 + 같은 설정 28항목 (piper_np4x), 돌아오기 6쌍 먼저
- 학습 때 메모리 확인 (단위 MemoryMax 26G — 넘으면 학습이 꺼짐)

## 10/23 12시
- **사용자 (12시): "셋 다 성공도 90 넘기자"** — 셋 다 = B 성공 + 돌아오기 + B 끝까지 유지 (돌아오기 6쌍 120장면). np2 + 고친 보조 78.3%, np3 + 고친 보조 87.5% (105/120: B 실패 9, 돌아오기 3, B 무너짐 2, 전환X 1)
- np3 의 B 실패 9 는 모두 B 블록을 못 쥠, 6 은 쥐는 높이에서 벌린 채 43~309스텝 → ⑲ `--stuck-grasp 40`. np2 + ⑲ (piper_np2fs, 8쌍까지): 셋 다 78.3 → 83.3, B 실패 18 → 12
- 돌고 있는 것: `piper-np3s` (run_np3s_round.sh): np3 + 고친 보조 + ⑲ (piper_np3fs), 돌아오기 6쌍 먼저 → 나머지 6쌍 → 혼자 10. `piper-k4b`: np2 + ⑲ 남은 줄 (piper_np2fs). `piper-np2h`: 지금 항목까지만 (np3 단독 8:5 등 진행 중)
- VLA 단독 평가 미룸: outputs/piper/deferred_alone.txt (np3 단독 8:4 1:8 4:1 + 혼자 10, np2 단독 12쌍 — claim_h3·claim_h2 자리표를 지우고 다시)
- piper-np2s 본체는 멈춤 (np2fs 성적표는 끝난 뒤 따로: board piper_np2fs)
- 14시: np3 + 고친 보조 + ⑲ (piper_np3fs) 돌아오기 6쌍 셋 다 105/120 = 87.5% — ⑲ 없을 때와 같음. ⑲ 는 켜졌지만 대신 쥐기가 실패
- diag_np3b (outputs/piper/diag_np3b.log): A8→B7 1012·1013 손목 각도를 못 읽는데(None) 예전 yaw_bad 가 남아 내려가기를 막고 막대 위 7~8cm 에서 머묾 → ⑨' `--yaw-none-reset 10`; A8→B5 1012 기억 1.7cm 틀림(사진은 0.2cm) 2.5cm 문턱에 걸림 → ⑰' `--mem-tol 0.012` (손이 블록 위 4cm 넘으면 손 밑이어도 고침); A8→B5 1013 VLA 가 책상 밖으로 헤맴·6번 관절 한계
- 시험 중: `piper-np3x` (run_np3x_test.sh): np3 + 고친 보조 + ⑲ + ⑨' + ⑰' + ⑳, A8→B5·A8→B7 20장면 (outputs/piper_np3x_switch) — np3fs 와 짝 비교
- np2 + ⑲ 혼자 과제도 미룸 (deferred_alone.txt)
- 15:10 새 옵션 시험 (piper_np3x, A8→B5·A8→B7): B 실패 8 → 3, 셋 다 32 → 35/40 (나빠진 0). 남은 실패 5 중 2 는 전환X (A 를 일 바꾸기 전에 끝냄)
- **지금 설정 후보 = np3 + 고친 보조 + ⑲ `--stuck-grasp 40` + ⑨' `--yaw-none-reset 10` + ⑰' `--mem-tol 0.012` + ⑳ `--open-empty`**
- 16시: np3 + 새 설정 6쌍 셋 다 104/120 (86.7%) — B 실패 9 → 5 지만 B 무너짐 3 → 5, 전환X 1 → 3. 전환X 빼면 104/117 = 88.9%
- diag_stack3 (outputs/piper/diag_stack3.log): B 무너짐 4장면 모두 돌아오기 중 빨강을 든 손이 쌓은 블록 3~6cm 옆을 z 0.94~0.96 로 지나다 침 → ㉒ `--stack-guard` (B 가 쌓기였으면 든 손이 쌓은 블록 10cm 안·13.5cm 아래면 먼저 들고 옆 0.3배), ㉑ `--stack-clear` (내려놓을 자리를 밑동·A 목표로 가는 선이 쌓일 자리 8cm 밖, 가설은 틀렸지만 같이)
- 17:45 ㉒ 결과 (piper_np3g): B 무너짐 5 → 0, 그러나 A4→B9 돌아오기 20 → 15 (회색 판이 쌓은 블록 9~12cm 라 놓으러 내려갈 때 계속 들어 올림) → 셋 다 31/40 그대로
- ㉒' `--guard-r 0.07`: 반경 7cm, 놓을 곳 5cm 안은 안 막음, 쌓은 블록 쪽 성분만 뺌. 시험 `piper-np3h` (run_np3h_test.sh → outputs/piper_np3h_switch)
- 시험 중: `piper-np3g` (run_np3g_test.sh): 새 설정 + ㉒ + ㉑, A8→B9·A4→B9 20장면 (outputs/piper_np3g_switch) — piper_np3x 와 짝 비교
- 돌고 있는 것: `piper-np3xr` (run_np3x_round.sh, log outputs/piper/np3x_round.log): 이 설정 28항목 → outputs/piper_np3x (8:5·8:7 은 시험 결과), 남은 돌아오기 4쌍 먼저. piper-np3s 는 지금 항목까지만 (자리표)
- ⑲ 로 못 살리는 np3 B 실패 3: ① 일 바꾸기 때 빨강 쥐기가 빗나가 빈 그리퍼가 닫힌 채 450스텝 (A8→B7 1007) → ⑳ `--open-empty` (빈손 다 닫힘·블록 4cm 밖이면 벌림) ② 막대 위 7~9cm 에서 400스텝 머묾 (A8→B7 1013) → `--stuck-dz 0.09` (⑲ 높이 한도) ③ 블록 위 5cm 에서 못 내려감 (A8→B5 1012, 손가락이 윗면에 걸림?) — 아직. 둘 다 기본 꺼짐, np3s 결과를 보고 시험

## 10/23 06시
- **목표 달성: 새 기준 np2 + 고친 보조 (piper_np2fd) 28항목 93.6% / 판 안 94.6%, 돌아오기 93.3% / 94.2%** (혼자 97.5, 일 바꾸기 90.4, 셋 다 78.3). 예전 보조 90.0 / 79.2 와 짝 비교 28항목 36:16 (p 0.008), 돌아오기 20:3 (p < 0.001). outputs/piper/score_piper_np2fd*.md, outputs/media/np2fd_compare.png (make_fig_np2fd.py)
- ⑲ `--stuck-grasp 40` 미리 보기 (piper_np2sgd, 노란 막대 두 쌍 40장면): B 30 → 34, 셋 다 28 → 31 (고친 보조 대비). 28항목 전체는 아직
- 일지 03_일지/2026-10-23.md 초안 (4K GIF 넣고 push 할 것)
- 돌고 있는 것: `piper-np2h` 의 np3 + 고친 보조 (piper_np3fd) → np3 단독 → np2 단독 (일꾼 3개; 3521690 은 4K 용으로 멈춤), `piper-k4` (run_k4_fd.sh): A4→B9 장면 1004 한국어 끼어들기 4K → outputs/media/k4_fix_1004.{mp4,gif}
- 07:22 4K 시연 성공 (outputs/media/k4_fix_1004.{mp4,gif}, 일지 10/23 6절, push 4176e93). 첫 시도는 실행기 끝 판정이 쌓기를 몰라(책상 식으로 위·아래 7cm) 맴돎 → `piper_sim/calib_front_stack.py` → front_calib_stack.json (위 블록·가린 아래 블록 식, 228 중 227 이 2.5cm 안), korean_live.CameraDone 은 쌓기면 이 식 + 마지막으로 벌린 손 높이 > 책상 5.5cm
- ⑲ 28항목 (piper_np2fs): `piper-np2s` (np2h 끝나면 일꾼 3개) + `piper-k4b` (시연 뒤 s2p 줄 이어받음, 일꾼 1개)
- 다음: ⑲ 28항목 결과, 쌓기 B 유지 (A4→B9 셋 다 14/20), np3 결과 비교, 4K 카메라를 조금 내려 앞쪽 판이 글자 칸에 안 가리게
- 09:30 np3 + 고친 보조 (piper_np3fd) 28항목 92.1% / 92.5%, 돌아오기 93.3 / 95.0, **셋 다 87.5%** (np2 78.3, 짝 22:11 p 0.08). 혼자 95.5·일 바꾸기 88.8 은 np2 보다 조금 낮음. 일지 10/23 7절 push (535489f)

## 10/23 03:35
- **⑯ 판 따라가기(--pad-track)는 쓰면 안 됨**: np2g 에서 A4→B5→A4 돌아오기 17 → 8/20. 실패 12장면 모두 회색 판은 그대로인데 ⑯ 이 판 기억을 7~29번 고침 (팔이 오른쪽 판을 가려 잘못 잼) → ⑤ 가 엉뚱한 자리 기준으로 놓기를 막아 빨강을 끝까지 듦
- **지금 설정 = BEST + `--mem-refresh --putdown-far` + `PIPER_FRONT_CALIB=piper_sim/front_calib_wide.json`** (+ `--release-wait 40`). korean_live.py 도 같음
- 돌고 있는 것: `piper-np2h` (run_np2h_round.sh, log outputs/piper/np2h_round.log): np2 + 이 설정 28항목 (piper_np2fd) → np3 (piper_np3fd) → np3 단독 (piper_np3mch) → np2 단독 (piper_np2mch). 일꾼 3개 + 시험 끝나면 1개
- `piper-sgtest` (run_sgd_test.sh): 이 설정 + ⑲ `--stuck-grasp 40`, A8→B7·A1→B7 20장면 (outputs/piper_np2sgd_switch)
- np2g(⑯ 포함)는 멈춤: piper_np2fb 에 8:0 8:3 8:5 8:7 8:9 4:5 만 (⑯ 때문에 쓰지 말 것)

## 10/23 02시
- 예전 보조 새 기준 28항목 (piper_np2mc, 02:29 끝): **90.0%** (혼자 98.5, 일 바꾸기 88.3, 돌아오기 **79.2**). outputs/piper/score_piper_np2mc.md
- 고친 보조 전부(⑩·⑮ 포함) vs 손 안 잡는 고침 — A8→B0·A8→B7 40장면 짝 비교 (B / 돌아오기 / 셋 다): 예전 34/29/27, 전부 29/34/29, **손 안 잡는 고침 33/36/32**
  → VLA 손을 직접 잡는 ⑩ 쥐기 확정·⑮ 먼저 들기가 B 를 떨어뜨림 (쥐어 주면 VLA 가 계속 벌리려 함). ⑮' 를 `--putdown-far` 로 따로 뺌
- **지금 가장 좋은 설정 = BEST + `--pad-track --mem-refresh --putdown-far` + `PIPER_FRONT_CALIB=piper_sim/front_calib_wide.json`** (+ 평가 `--release-wait 40`)
- 돌고 있는 것: `piper-np2g` (run_np2g_round.sh, log outputs/piper/np2g_round.log, 일꾼 4개): np2 + 손 안 잡는 고침 28항목 (piper_np2fb; 8:0·8:7 은 run_fixb_test.sh 결과) → np3 + 같은 설정 (piper_np3fb) → np3 단독 (piper_np3mch) → np2 단독 일 바꾸기 (piper_np2mch). piper-np2f(고친 보조 전부)는 멈춤 (A8→B0·B3·B5·B7 만 piper_np2fc 에)
- 다음: np2fb 28항목 → 짝 비교 → 일지 10/23 + 4K GIF (korean_live 보조 목록을 이 설정으로) → 남은 실패 진단 (노란 막대 B 가 가장 큼: A8→B7 B 15/20)
- 03시: 노란 막대 B 실패 5 중 4 는 막대를 한 번도 못 쥐고 쥐는 높이에서 벌린 채 67~231스텝 머묾 → ⑲ `--stuck-grasp 40` (누적 2초 머물면 ⑥ 처럼 내려가 쥐고 5cm 듦). `piper-sgtest` (run_sg_test.sh): 손 안 잡는 고침 + ⑲, A8→B7·A1→B7 20장면 (outputs/piper_np2sg_switch). np2g 일꾼 3487612 멈춰 비운 자리 (np2g 는 일꾼 3개)
- 짝 비교: `python cmp_rounds.py outputs/piper_np2mc outputs/piper_np2fb --names 예전 고침` (28항목·돌아오기·셋 다, 부호 검정)
- korean_live.py 보조 목록·변환식을 손 안 잡는 고침으로 바꿈 (4K GIF 용)

## 10/22 23시
- **17:04 systemd-oomd 가 Claude 앱 단위째 꺼서 nohup 으로 띄운 평가까지 죽었다 (memory oom-separate-unit).** 22:32 다시 시작. 이제 긴 스크립트는 `systemd-run --user --unit=이름` 으로 (상태 `systemctl --user status 이름`, 끄기 `systemctl --user stop 이름`)
- 돌고 있는 것: `piper-np3c` (run_np3c_round.sh, log outputs/piper/np3c_round.log): 새 기준 np2 + 보조 (piper_np2mc) → np3 + 보조 → np3 단독 → np2 단독. 일꾼 4개 중 1개(3295053)는 진단용으로 멈춤 → 3개로 돎
- 사용자 (22:50): "그동안 돌아오기 실패 원인 찾아서 고쳐줘". 새 기준 np2 + 보조 돌아오기: A8→B0 75%, A8→B5 85%, A8→B7 70% (B 75%)
- 진단 (`diag_pad.py`, 같은 장면 다시 돌리며 판·기억·손목 상태 기록, log outputs/piper/diag_pad.log):
  ① 블록별 변환식은 처음 자리 근처로만 맞춰, 새 빈 자리에 내려놓은 빨강을 2.2~3.2cm 틀리게 잼 → 정렬 보조가 블록 옆을 짚음 (넓은 변환식이면 평균 0.28cm)
  ② 다시 쥔 직후 놓기 보조(④)가 책상 높이 손을 판 쪽으로 끌어 블록이 판을 6cm 밂 (A4→B9 1000) — 빈 자리가 목표 판에서 9.5cm
  ③ 헛쥐기로 블록이 밀렸는데 기억(손 8cm 안이면 안 고침)이 옛 자리 → 손을 빈 자리에 400스텝 붙잡음 (A4→B9 1004)
  ④ 쥐는 높이에서 정렬도 맞았는데 VLA 가 안 오므리고 다시 듦 (노란 막대 B, A8→B7 1005·1006·1009; 돌아오기 A4→B9 1000)
- 고침 (모두 켜기 옵션, 끄면 예전과 같음): grasp_assist ⑮ `--pad-safe` (책상 높이면 먼저 듦 + 빈 자리는 목표 판에서 13cm 밖), ⑯ `--pad-track` (밀린 판 기억 고침), ⑰ `--mem-refresh` (손 8cm 안이어도 계속 다르게 보이면 고침) + 넓은 변환식 `PIPER_FRONT_CALIB=piper_sim/front_calib_wide.json` + 쥐기 확정 `--close-assist`(⑩, 예전 것)
- 시험 중: `piper-fixtest` (run_fix_test.sh, log outputs/piper/fix_test.log, 결과 outputs/piper_fix_test/e장면/): 실패 20장면을 같은 장면으로. 첫 3장면 2개 살아남. 채점 = scratchpad fixscore.py 같은 방식 (전: piper_np2mc_switch·piper_release_test)
- 실패 20장면 결과: 돌아오기 4 → 17, 셋 다 0 → 14 (일지 10/22 12절, outputs/media/padpush_fix.png)
- 23:38~ `piper-np2f` (run_np2f_round.sh, log outputs/piper/np2f_round.log): np2 + 고친 보조 28항목 (piper_np2fc) → np3 + 고친 보조 (piper_np3fc) → np3 단독 (piper_np3mch) → np2 단독 (piper_np2mch). np3c 의 np3·단독 줄은 자리표를 미리 만들어 건너뜀. 일꾼 1개, np3c(3295048) 끝나면 3개 더
- 10/23 01시: 고친 보조 A8→B0 20장면 짝 비교: 돌아오기 75 → 90, B 95 → 80 (셋 다 15 → 16). B 가 나빠진 장면은 ⑮ 먼저 들기가 45스텝 넘게 켜짐 (그런 장면 B 2/7, 아니면 29/31). diag_held.py: ⑩ 이 대신 쥐어 주면 VLA 가 계속 벌리려 하고 ⑤ 가 막아 헤맴 (A8→B0 1010)
- 시도: ⑩' `--close-wait 12` (VLA 스스로 쥐기 기다림), ⑱ `--regrasp-ok` (쥔 자리에서 다시 쥐기 허용) — 몇 장면으로는 차이 안 보임 (run_fix2_test.sh, run_fix3_test.sh 중단)
- 지금: `piper-fixbtest` (run_fixb_test.sh): VLA 손을 잡는 ⑩·⑮ 빼고 넓은 변환식 + ⑯ + ⑰ + ⑮' `--putdown-far` 만, A8→B0·A8→B7 20장면 (outputs/piper_np2fb_switch) — 예전 보조(piper_np2mc)·고친 보조(piper_np2fc)와 짝 비교
- 일꾼 3296358 도 멈춤 (np3c 는 일꾼 2개). GPU 4개 = np3c 2 + np2f 1 + 시험 1

## 10/22 15시
- **사용자: "이제 그 기준으로 하자" → 새 기준 = 새 지시를 들고 옮기는 도중(grasp:30)에, 쥔 블록은 가까운 빈 자리에 (memory research-scenario). 예전 grasp:3 점수와 섞지 않는다**
- **사용자: "일지 쓸때마다 정확도 높은걸로 gif 4k 화질로"** (memory gif-4k): `render4k.py` (자유 시점 3840×2160 + 손목 720 + 한국어 글), `korean_live.py --render4k 이름 --title ...` → mp4 + GIF(100MB 넘으면 장 수 줄임)
- 지금 가장 좋은 보조 = `--approach-assist --descend-assist --unjam-assist --yaw-assist --putdown-assist --fresh-held --putdown-near`
- 새 기준 미리 보기 (`run_midcarry_test.sh`, 돌아오기 6쌍 × 5): 새 지시 27/30, 돌아오기 24/30, 둘 다 21/30
- 발견: 쌓기 B(B9)는 거의 끝까지 남지 않음 (예전 기준도 40 중 1) — 블록이 닿는 순간 B 성공 → 아직 쥔 채 돌아오기로 → 내려놓기 보조가 그 블록을 빈 자리로 (diag_stack.py)
- 고침: 평가 `--release-wait 40` (B 성공 뒤 손 벌릴 때까지, 새 기준에 넣음) + 보조 ⑭ `--wrong-guard` (다른 블록 위에서 못 오므림, 같은 지시 아래 엉뚱한 블록이면 쥔 자리에 도로). 4쌍 × 5: 셋 다(B·돌아오기·B 유지) 9 → 14 /20
- 돌고 있는 것: `run_np3c_round.sh` (PID 3109070, log outputs/piper/np3c_round.log): np3 학습(3047081) 끝 → 새 기준 np2 + 보조 (piper_np2mc) → np3 + 보조 (piper_np3mc) → np3 단독 (piper_np3mch) → np2 단독 일 바꾸기 (piper_np2mch). np3b·np3 본체는 끔
- `run_midcarry_test.sh` (PID 3068267): 새 기준 미리 보기, 돌아오기 6쌍 × 5장면 (outputs/piper_midcarry_test)

## 10/22 11시
- 사용자 (14시): "그렇게 하자 돌아오기 90 넘기자" — 한국어 끼어들기 명령을 믿고 쓸 기준. 지금 돌아오기 85.8% (6쌍 120번 중 103), 끼어들기 한 문장 끝까지 84.2%
- 원인 하나: 돌아오기에서 원래 블록을 제대로 쥐었는데 보조 기억 속 '유령' 자리(공중의 막대를 잘못 읽음) 때문에 다른 블록으로 보고 내려놓기가 빈 자리로 가져감 (diag_putdown.py, A1→B7 1001)
- 고침: grasp_assist ⑬ `--fresh-held` (쥐는 순간 지금 사진 위치, 쥔 블록 기억 안 고침, 내려놓기는 지시문 블록이 책상에 보일 때만). 실패 장면 다시 돌리기 (`run_freshheld_test.sh`): 돌아오기 실패 9 중 4 성공으로, 성공 장면 그대로
- 사용자 (14:50): "어느 정도 수행했다가 다른 곳에 놨다가 다른 태스크하고 다시" → `--putdown-near` (손에서 가장 가까운 빈 자리), `run_midcarry_test.sh` (PID 3068267): 쥔 뒤 30스텝에 새 지시, 돌아오기 6쌍 × 5장면 (outputs/piper_midcarry_test)
- 남은 실패(np2pd 35/560)는 여러 가지로 흩어짐 → VLA 단독(25.5%) 올리기로: 보조 증류
- `assist_collect.py`: np2 + 보조 5가지를 평가와 똑같이 돌리며 성공 구간의 (관측, 보조까지 거친 동작)을 시범으로 저장 (DA… 파일 = 교정 시범 몫)
- 돌고 있는 것: `run_np3_round.sh` (PID 2913426, log outputs/piper/np3_round.log): 보조 시범 (장면 3300~3319 × 22) → 정리 → np3 학습 (np2 에서, 10000) → np3 + 보조 5가지 (piper_np3pd) → np3 단독 (piper_np3h)

## 10/22 08시
- 한국어 명령 시작 (사용자 규칙 90% 넘음): `korean_cmd.py` (규칙 해석기: 할 일·끼어들기·돌아오기·되묻기·멈춤), `korean_live.py` (시뮬 실행기: 대본 `--script "0=..." "grasp+3=..."` 또는 터미널 입력, `--show` 창, `--video`). 끝 판정은 정면 카메라 + 그리퍼만 (판 안 4.5cm / 블록 위 2.5cm, 10스텝)
- 시연 2장면 성공: 1002 (빨강→보라, 끼어들기 초록→흰 → 돌아옴), 1005 (빨강→주황, 끼어들기 노란 막대→흰 → 돌아옴). outputs/media/korean_demo_100{2,5}.{mp4,gif}
- 정면 카메라 변환식을 책상 전체로 다시 맞춤 (`piper_sim/calib_front_wide.py` → front_calib_wide.json, 블록 공통, 평균 0.28cm; 예전 식은 블록이 제자리를 떠나면 평균 1.8·최대 13.6cm 틀림). 보조는 `PIPER_FRONT_CALIB=piper_sim/front_calib_wide.json` 이면 새 식
- 10:19 np2 + 보조 5가지 + 새 변환식 (piper_np2pw) 28항목 93.0 / 93.9 — 예전 식 93.8 / 94.6 과 짝 비교 20:24 (p 0.65), 차이 없음 → 보조는 예전 식 그대로, 새 식은 한국어 실행기 끝 판정만
- 돌고 있는 것 없음
- GPU 를 잠깐 쓸 때: outputs/piper/gpu_launch.lock 을 mkdir 로 잡고 쓰고 rmdir (일꾼들이 같은 자물쇠를 씀)
- 09:06 '저거' 가리키기 추가 (korean_cmd.resolve_pointing, Controller.click, korean_live 창 누르기·대본 "click:x,y"). 시연 korean_point_1002 성공

## 10/22 07시
- **np2 + 보조 + 막힘 풀기·손목 맞추기 + 내려놓기 (piper_np2pd) 28항목: 3cm 93.8% / 판 안 94.6%** (혼자 97.5, 일 바꾸기 94.6, 돌아오기 85.8), 95% 넘는 항목 20개. np2uj 대비 짝 비교 56:15 (p 1e-6)
- 기본 보조 = `--approach-assist --descend-assist --unjam-assist --yaw-assist --putdown-assist`
- VLA 단독 np2 25.5 / 36.4 (np1 과 같음)
- **사용자 규칙 "90% 넘으면 해석기 시작하자" → 넘음. 다음: 한국어 명령 해석기 + 지시 입력 창** (memory korean-command-plan)
- 돌고 있는 것 없음 (GPU 비었음)

## 10/22 02시
- 06:25 중간 (np2pd 6/22): 같은 6쌍에서 np2uj 대비 새 지시 98 → 116 /120, 돌아오기 4쌍 54 → 73 /80 (A4→B5·A8→B5 는 새 지시·돌아오기 모두 20/20)
- np2 단독 28항목 (05:09 끝): 3cm 25.5% / 판 안 36.4% (혼자 36.5, 일 바꾸기 25.4, 돌아오기 7.5) — VLA 혼자는 np1(25.5/39.3)과 같음. run_np2c_round.sh 끝, run_np2d_round.sh 05:09 시작 (piper_np2pd)
- 사용자: "90% 넘으면 해석기 시작하자" (VLA + 보조 28항목 3cm 평균 기준, memory korean-command-plan)
- 발견: np2uj 일 바꾸기 140장면 중 59장면은 VLA 가 A 를 B 의 놓을 곳에 내려놓음 → B 성공 32/59 (다른 곳 78/81), 돌아오기 17/47 (다른 곳 67/73). 시범은 A 를 쥔 자리에 그대로 내려놓는다
- 고침: grasp_assist ⑫ 내려놓기 `--putdown-assist` (쥔 블록이 지시문의 블록이 아니면 빈 자리 — 처음 자리 우선, 다른 물체 9cm·B 블록·놓을 곳 12cm — 로 옮겨 내려놓음). 가짜 로봇 시험 통과 (74스텝)
- 걸어 둔 것: `run_np2d_round.sh` (PID 2510324, log outputs/piper/np2d_round.log): run_np2c_round.sh 끝나면 np2 + 보조 + 막힘 풀기·손목 맞추기 + 내려놓기 28항목 (piper_np2pd)

## 10/22 01시
- **np2 + 보조 + 막힘 풀기·손목 맞추기 (piper_np2uj) 28항목: 3cm 86.4% / 판 안 88.4%** (혼자 97.0, 일 바꾸기 85.8, 돌아오기 70.0), 95% 넘는 항목 14개. 같은 np2 + 기존 보조 74.1/77.3 대비 짝 비교 86:17 (p 3e-12). 이제 기본 보조 = `--approach-assist --descend-assist --unjam-assist --yaw-assist`
- 남은 실패: 돌아오기 '블록 위에서 못 쥠' 22, 새 지시 '판 밖에 놓음' 13 (fail_piper_np2uj.md)
- 돌고 있는 것: run_np2c_round.sh 의 마지막 np2 단독 28항목 (piper_np2h) → 끝나면 점수표·실패 나누기 자동
- 그 뒤 할 것: 돌아오기 못 쥠 장면 diag_hover 로 다시 돌리기 (np2 + 새 보조), 판 밖 놓기 원인, 새 보조 켠 교정 시범


## 10/21 18시 (가장 최신, 이것부터)
- **np2 + 보조 28항목 (22:40 끝): 3cm 74.1% / 판 안 77.3%** (혼자 94.5, 일 바꾸기 73.8, 돌아오기 40.8), 95% 넘는 항목 10개. np1 + 보조 71.8/74.3 보다 높음. 실패는 여전히 '블록 위에서 못 쥠' (B 21, 돌아오기 32) — fail_piper_np2apr.md
- 22:15~ np2 + 보조 + 막힘 풀기 + 손목 맞추기 (piper_np2uj) 평가 중, 그 뒤 np2 단독 (piper_np2h)
- 23:43 중간 (np2uj 8/22 실행): 같은 8쌍에서 np2 + 보조 대비 새 지시 104 → 130 /160, 돌아오기 6쌍 49 → 84 /120. 혼자 하는 과제(망가지는지)는 아직
- 사용자: "일단 정확도 먼저 높이자" (실물 지시 창·한국어 해석기는 나중)
- 발견 (`diag_hover.py`, np1 + 보조 '블록 위에서 못 쥠' 8장면 다시 돌림): 손가락이 블록 윗면·모서리나 책상에 걸림 (막대는 손가락 방향이 66~76° 어긋나 윗면에 얹힘) → 내려가기 재촉(⑦)이 계속 누름 → 손이 70° 넘게 기울며 4·5·6번 관절이 한계에 박혀 끝까지 멈춤. 보조가 기억한 블록 위치가 밀린 블록을 모르고 2.4cm 옆을 누른 장면도 있음
- 고침 (grasp_assist): ⑪ 막힘 풀기 `--unjam-assist` (멈춤 1초 + 손목 관절 한계 3° 안 → 7cm 위로 빼며 처음 자세 손 방향, 블록 위치 새로 잼), ⑨ 손목 맞추기는 막힘 풀기와 같이 켜면 6번 관절 한계를 봄 (대칭 반대 방향)
- 실패 장면 그대로 다시 돌리기 (`run_unjam_test.sh`, 9장면의 실패 16번): 막힘 풀기만 4번, 막힘 풀기 + 손목 맞추기 10번이 성공으로 바뀜, 원래 성공은 그대로. 원래 성공하던 장면이 망가지는지는 28항목으로 재는 중
- 18:40 본체를 `run_np2c_round.sh` 로 바꿈 (PID 2275043, log `outputs/piper/np2c_round.log`; 학습 2174024 은 그대로, 학습 로그는 np2_round.log): np2 학습 끝 → 일꾼 4개: np2 + 보조 (piper_np2apr) → np2 + 보조 + 막힘 풀기 + 손목 맞추기 (piper_np2uj) → np2 단독 (piper_np2h)
- np2 학습 끝 19:14 (val_loss 2500 0.0633 → 5000 0.0616 → 7500 0.0602 → 10000 0.0603), merged 에 relfeat·sufstate 표시 있음. 19:14~ 일꾼 4개 평가 중


## 10/21 08시 (가장 최신, 이것부터)
- 받침대 없음 np1 + 보조 28항목: 3cm 71.8% / 판 안 74.3% (혼자 92.0, 일 바꾸기 73.3, 돌아오기 35.0), 95% 넘는 항목 7개 — 받침대 장면 최고(61.4%)보다 높음 (장면이 달라 같은 비교 아님)
- np1 단독 28항목 (12:03): 3cm 25.5% / 판 안 39.3% (혼자 34.0, 일 바꾸기 28.3, 돌아오기 5.8)
- 돌고 있는 것: `run_np2_round.sh` (12:03~) 교정 시범·돌아오기 시범 → np2
- 걸어 둔 것: `run_np2_round.sh` (log `outputs/piper/np2_round.log`): np1 단독 평가 끝 → np1 교정 시범 (보조 켬, PIPER_NEAR_GRASP=1) ‖ 돌아오기 쌍 시범 더 (data/piper_np_switch2) → np2 학습 (np1 에서) → np2 + 보조 → np2 단독

## 10/20 17시 — 받침대 없이 (가장 최신, 이것부터)
- 사용자: "앞으로 바침대 없이해줘" → 모든 새 작업은 `PIPER_NOPED=1` (memory piper-no-pedestal). layout_noped.json (밑동 −0.33,0,책상 높이), assets/robots/piper_noped, init_qpos_noped.json (책상 위 18cm·앞으로 45°), task_experts NOPED (쥐는 방향 똑바로 아래 기준 ±35°, 막대 2.8cm), grasp_assist 막대 높이 0.028
- 01:45 본체 `run_np_round_d.sh` (사진 특징 여러 줄 충돌로 한 줄 죽음 → 남은 줄 끝나면 한 줄로 빠진 것 채우고 학습)
- 22:45 본체 `run_np_round_c.sh` (일 바꾸기도 미리 계산 np_pre2.log, 회복만 다듬음)
- 21:45 본체를 `run_np_round_b.sh` 로 바꿈 (수집기 1063228·1142252 와 정상·넓은 배치 미리 계산 np_pre.log 1313996 을 기다리고 일 바꾸기·회복만 다듬음)
- 돌고 있는 것: `run_np_round.sh` (log `outputs/piper/np_round.log`): 받침대 없음 시범 수집 (data/piper_np_*) → 다듬기·검출 → find_rise (exclude_rise_np.txt) → 받침대 als3b 평가(1044443) 끝나면 사진 특징 → np1 학습 (als2 에서, --renorm-action, 15000) → piper_np1apr → piper_np1h
- als3b 6항목 (20:04): 혼자 30/40, B 37/80, 돌아오기 15/60 (als2 37·28·21) — 합계 비슷, 받침대 실험 끝
- 18:50 train_lora 버그 (relfeat 표시가 SUBTASK elif 때문에 지워짐) 고침. als3b 폴더에 relfeat 표시 붙이고 `run_near_eval2.sh` 로 6항목 다시
- 받침대 실험 마무리: als3b 학습 (1018500) → `run_near_eval.sh` 6항목 (T0·T8·A8→B0·B5·B7·B9, piper_als3bapr)
- 받침대 있는 성적(als2 + 보조 61.4% 등)과 받침대 없는 성적은 섞지 않는다

## 10/20 14시 — 다시 시작 (가장 최신, 이것부터)
- 사용자: "자 다시 해봐" (그만하라 할 때까지)
- 발견: '쥐는 높이에서 손을 올리는' 버릇은 교정·회복 시범에서 옴 — 시범 프로그램이 블록 바로 위에서 넘겨받아도 8cm 위로 먼저 올라감 (75개 중 63개). 정상·일 바꾸기 시범에는 없음
- 고침: `task_experts._near_grasp` (PIPER_NEAR_GRASP=1), `find_rise.py` → outputs/piper/exclude_rise.txt (111개), `train_lora.py --exclude`
- 돌고 있는 것: `run_near_round.sh` (log `outputs/piper/near_round.log`): 고친 시범 프로그램으로 보조 켠 교정 시범 (data/piper_al_dagger_als2n, dagger_scale 0.75) → 다시 거름 → als3b 학습 → piper_als3bapr → piper_als3bh
- 비교: als2 + 보조 61.4/64.3 (가장 좋음), als3a + 보조 57.5, sub2 + 보조 55.0

## 10/20 07시 — 쉼 (가장 최신, 이것부터)
- 사용자 "내일 아침 9시까지만" → sub2 + 보조 28항목까지 끝내고 멈춤. 돌고 있는 것 없음
- 결과 (모두 정렬·놓기·내려가기·다가가기 보조, 3cm/판 안): als2 61.4/64.3 (가장 좋음), als3a 57.5/59.8, sub1 48.2/49.3, sub2 55.0/59.5. VLA 단독 als2 20.7/35.7
- 단계 지시는 효과 없음 (sub2 ≈ als3a). '블록 위에서 못 쥠' 은 줄었지만 A 집기 실패(전환 안 됨) 9 → 26
- 고친 것: 단계 지시 모델은 지시문 48토큰 채움 (CUDA 그래프 메모리), Sequencer 쥔 블록 판정은 쥐는 순간 지금 사진 위치
- 다시 시작할 때 의논할 것: 학생 모델 쪽 기본기 (데이터 양·학습 길이), 못 쥠 장면 영상 분석, 항목당 50장면 재기

## 10/19 23시 — 다시 시작 (가장 최신, 이것부터)
- 사용자: "내일 아침 9시까지만 하자" → 10/20 09:00 까지. 08:47 마무리 예약 (결과 적고, 안 끝난 평가는 9시에 끔)
- 돌고 있는 것: `run_sub2.sh` (log `outputs/piper/sub2.log`): sub2 학습 (als2 에서, `--subtask tag` = 원래 지시문을 살린 단계 지시) → sub2 + 보조 28항목 (piper_sub2apr) → 점수표·fail_piper_sub2apr.md
- 비교: als2 + 보조 61.4/64.3 (가장 좋음), als3a + 보조 57.5, sub1 + 보조 48.2
- 03시: 쥔 블록 판정 고침 (subtask.Sequencer: 쥐는 순간 지금 사진 위치) → `run_sub2_eval.sh` 로 sub2 + 보조 28항목 처음부터 (예전 3항목은 outputs/old_sub2apr_before_fresh)
- 02시: 평가 프로세스가 지시문 길이마다 CUDA 그래프를 따로 만들어 2.9GB 까지 커짐 → 일꾼 하나 끔(8:3 다시), 지시문 48토큰 채움(switch_experiment, 단계 지시 모델 + CUDA 그래프일 때만). `run_sub2_extra.sh` 가 예전 코드 평가 하나가 끝나면 일꾼으로 들어감

## 10/17 21시 — 쉼 (가장 최신, 이것부터)
- 사용자: "지금 하는데 까지 하고 쉬자". `run_sub_round.sh` 의 sub1 + 보조 28항목(piper_sub1apr)만 끝까지. sub1h·als3ah 는 claim 을 미리 만들어 건너뜀 (outputs/piper/claim_s1h, claim_a3h)
- als3a + 보조 28항목: 57.5% / 59.8% (als2 + 보조 61.4% / 64.3% 보다 낮음 — 보조 켠 교정 시범 효과 없음)
- sub1 + 보조 결과 (10/18 00:48): 48.2% / 49.3% — als2 + 보조 61.4% 보다 낮음. 집기 단계가 나빠짐 (전환 안 됨 52/240). 가설: 'pick up X' 에 놓을 곳이 빠져 쥐는 방향이 여러 개가 됨 → 다음엔 원래 지시문 + 단계 표시 ("[pick] put the red block on ...")
- 지금 가장 좋은 것: als2 + 정렬·놓기·내려가기·다가가기 61.4% / 64.3% (VLA 단독 als2 20.7% / 35.7%)
- 다시 시작할 때 (아래 1·2 는 끝남)
  1. piper_sub1apr 결과 확인 (compare_runs.py piper_als2apr piper_sub1apr, fail_classify.py outputs/piper_sub1apr, subtask_log 로 단계 전환 확인)
  2. 단계 지시가 효과 있으면: sub1 단계 지시만(보조 없이) 28항목 — claim_s1h 지우고 worker 다시. 실패 나누기로 남은 실패 보기
  3. 효과 없으면: 단계 지시 데이터 늘리기(put it down 이 4,318 프레임뿐) 또는 다른 방향 의논

## 10/17 15시 (가장 최신, 이것부터)
- 사용자와 정함: 단계 나누기 (`subtask.py`: pick up X / put it on Y / put it down). 학습 `--subtask` (.subtask.npy, 3,427개에 붙임), 평가는 모델 폴더 `subtask` 표시면 자동 (Sequencer)
- 돌고 있는 것: als3a 학습 (2284901, 본체 run_asd_round.sh 는 끔) → `run_sub_round.sh` (log `outputs/piper/sub_round.log`): sub1 학습 (als2 에서, 같은 데이터 + data/piper_al_dagger_als2a_t) → 일꾼 4개: piper_als3aapr → piper_sub1apr → piper_sub1h → piper_als3ah
- 비교 기준: als2 단독 20.7/35.7, als2 + 정렬·놓기 52.0/57.0, + 내려가기·다가가기 (piper_als2apr) 61.4/64.3
- 사용자에게 한 예상: 70~80% 며칠, 90%+ 는 단계 나누기가 효과 있으면 1~2주, 95% 확인은 항목당 50~100장면으로 다시 재야 함

## 10/17 11시 (가장 최신, 이것부터)
- 결과: als2 + 정렬·놓기·내려가기·다가가기 28항목 3cm 61.4% (혼자 83.5 / 바꾸기 56.7 / 돌아오기 34.2), 판 안 64.3%. 단독 20.7%, 정렬·놓기 52.0%
- 남은 가장 큰 실패 '블록 위에서 못 쥠' = 손가락이 블록 면과 18° 어긋남 (쥔 장면 4°) → 손목 맞추기 `--yaw-assist` (grasp_assist ⑨)
- 손목 맞추기는 B 못 쥠 35→22 지만 전환 안 됨 0→12 → 끔. 못 쥠의 진짜 모습: 쥐는 높이에서 학생이 손을 올림, 쥐어 줘도(--close-assist) 안 들어 올림 → 학습으로
- 돌고 있는 것: `run_asd_round.sh` (log `outputs/piper/asd_round.log`): yaw 시험 끝 → 보조 켠 교정 시범 (dagger --approach-assist --descend-assist, data/piper_al_dagger_als2a) → als3a 학습 → als3a+보조 28 → als3a 단독 28
- (끝남) `run_yaw_test.sh` (출력 piper_als2yaw): 7항목. 비교 기준 piper_als2apr 같은 항목 (`compare_runs.py piper_als2apr piper_als2yaw`, `fail_classify.py`)
- 다음: 효과 있으면 28항목 전체 (--approach-assist --descend-assist --yaw-assist). 보조가 손댄 비율을 같이 적기

## 10/17 0시 (가장 최신, 이것부터)
- 돌고 있는 것: alc2 학습 (1568264, 고아 서브셸 1568262 — 본체 run_alc2_round.sh 는 끔) → `run_alc3_probe.sh` (log `outputs/piper/alc2_round.log`): alc2 끝 → alc2 넓은 민감도 (`sens_wide.py`, 빨강 +5cm) → alc3p 학습 (alc2 에서, `--cam-drop 0.4 --cam-drop-both 0.2 --new-lr 1e-3`, 2500스텝) → 민감도 → 일꾼 4개: als2 + 정렬·놓기 보조 28항목
- alc2 는 새 층 가중치가 3~4배 자랐는데도 민감도 0.2cm 안 → 학습률 문제 아님. 손목 카메라에 블록이 보여서라는 가설로 두 카메라 다 가리기
- alc3p 결과 (01:33): 카메라가 보이면 0.25cm 안 → 위치 입력 방향은 접음. 지금 일꾼 4개가 als2 + 보조 28항목 (01:37~, 아침 7~8시). 다음은 보조 켠 실패 나누기
- 01:45 `run_assist2_test.sh` (log `outputs/piper/assist2_test.log`) 가 alc2_round.log 의 '끝' 을 기다림 → als2 + 정렬·놓기 + 내려가기 재촉(`--descend-assist`) / + 다가가기(`--approach-assist`) 6항목. 실패 나누기 `fail_classify.py outputs/piper_als2place` 로 비교. 평가 기록에 `assist_steps` (보조가 손댄 스텝 수) 추가
- als2 + 정렬·놓기 28항목 (07:15): 3cm 52.0% (혼자 81.5 / 바꾸기 43.3 / 돌아오기 20.0), 판 안 57.0%. 내려가기만으로는 효과 없음, 다가가기가 효과. 다가가기까지 켜면 보조가 앞뒤좌우를 정하는 스텝이 절반 넘음 (논문에 비율 같이 적기)
- 05:40 `run_apr28.sh` (log `outputs/piper/apr28.log`): run_alc3_probe.sh·run_assist2_test.sh 가 끝나면 als2 + 정렬·놓기·내려가기·다가가기 28항목 (이미 잰 6항목 건너뜀) → 점수표 → 실패 나누기 md 3개. 첫 결과: 일 바꾸기 4항목 B 20 → 37/80, 돌아오기 13 → 28/80
- GIF: `run_gif_rec.sh` + `make_gif_1017.py` → 10/16 일지 6절 (보조 비교 2개)

## 10/16 23시 (가장 최신, 이것부터)
- 돌고 있는 것: `run_alc2_round.sh` (log `outputs/piper/alc2_round.log`; run_alc_round_d.sh 는 끔): alc1 평가 4개(쌍 8:0·8:3·8:5·8:7) 끝 → alc2 학습 (alc1 에서, `--new-lr 1e-3 --cam-drop 0.3`, 2500스텝마다 저장) ‖ 2500스텝 민감도 → alc2 민감도 → 일꾼 4개: alc2 28항목 → als2 + 정렬·놓기 보조 28항목
- 발견: alc1 (정면 카메라 가리기 30%) 도 민감도 0.1cm — 정면 카메라를 실제로 가려도 그대로. 새 층 state_suffix_proj 가중치가 3e-5 로는 거의 안 자람 (열 노름 als1 0.05 → als2 0.06 → alc1 0.08)
- 새 것: `train_lora.py --new-lr` (state_suffix_proj 만 다른 학습률, OneCycle 묶음별)
- als2 결과: 3cm 20.7%, 판 안 35.7% (als1 과 같음)

## 10/16 16시 (가장 최신, 이것부터)
- 돌고 있는 것: als2 28항목 평가 서브셸 4개 (본체 run_als_round_c.sh 979006 은 16:40 에 끔 — als3·als4 단계 취소) →
  `run_alc_round_d.sh` (20:50, log `outputs/piper/alc_round.log`; a·b·c 는 끔, b 가 18:47 에 띄운 alc1 학습 1423186 은 그대로): als2 평가 끝 → 점수표 → alc1 학습 혼자 끝 (평가 옆에서 2.7배 느림) → 민감도 → 일꾼 4개가 alc1 28항목 → als2 + 보조 28항목 (19:43 겹쳐서 OOM 난 뒤 모든 평가는 자물쇠 + 메모리 확인). alc1 학습 (als2 에서, `--cam-drop 0.3` 정면 카메라 가리기) ‖ als2 + 정렬·놓기 보조 28항목 (2개) → alc1 민감도 → alc1 28항목 (일꾼이 mkdir 로 항목을 나눠 가짐, GPU 남은 메모리 2.8GB 넘을 때만 띄움)
- 새 것: `--cam-drop` (train_lora → ttrtc.STATE["cam_drop"]): 학습 장면 일부에서 정면 카메라 사진 토큰을 주의 마스크로 가림. 평가에는 영향 없음 (CPU 확인)
- 보조 성적 (als1, T0·T8·A8→B5·A4→B9 120측정): 단독 21.7%/36.7% → 정렬·놓기 보조 50.8%/56.7%. 쥐기 마무리(`--grasp-complete`)는 더한 것 없음 (47.5%/51.7%)
- als2 민감도: 여전히 0.1cm 남짓 (입력 무시). sufstate 로도 학생이 손 기준 위치를 안 씀
- als2 결과 (21:26): 3cm 20.7% (혼자 33.5 / 바꾸기 16.7 / 돌아오기 7.5), 판 안 35.7% — als1 과 같음. 교정 시범만으로는 안 오름
- 할 것: 가장 좋은 모델로 28항목 전체 + `--place-assist` (VLA + 보조 전체 성적). 남은 실패는 일 바꾸기 B·돌아오기

## 10/15 23시 — 상태를 동작 쪽에도 (가장 최신, 이것부터)
- 발견: alr1 은 손 기준 물체 위치를 2cm 바꿔 넣어도 동작이 0.1cm 도 안 바뀐다 (`sens_check.py`, CPU). SmolVLA 는 상태를 얼린 VLM 앞부분에 토큰 하나로만 넣기 때문
- 고침: `ttrtc.enable_sufstate()` — 정규화된 상태 32칸을 선형층(`state_suffix_proj`, 처음 0)으로 동작 토큰마다 더함 (π0 처럼). 학습 `--sufstate`, 모델 폴더 `sufstate` 표시면 평가·교정 시범이 자동으로 켬. 확인: 처음 출력 차이 0.0, 기울기 흐름
- 돌고 있는 것: alr1 28항목 평가 서브셸 4개 (본체 run_alr_round_b.sh 는 끔, log `alr_round.log`) → `run_als_round.sh` (log `als_round.log`): alr1 점수표 → al1 교정 시범 135개 정리 → als1 학습(GPU 혼자) → 민감도 → 28항목 → als 교정 시범(절반 규모, `outputs/piper/dagger_scale.txt`=0.5) → als2 → als3 → als4
- 끈 것: alr1 교정 시범·alr2 (run_alr_round_b.sh 뒤 단계), alr1+보조 평가 (run_alr_assist.sh)
- alr1 결과: 3cm 15.5%, 판 안 28.7% (al1 과 같음). als1 도 민감도 0.1cm (입력 무시) — 10/16 04시
- als1 결과 (10/16 09시): 3cm 19.6% (혼자 29.0 / 바꾸기 17.9 / 돌아오기 7.5), 판 안 35.7% — VLA 단독 최고
- 놓기 보조 `--place-assist` (grasp_assist.py place=True): `run_place_test.sh` 가 als1 평가 뒤 al1 + 정렬 + 놓기 보조 8항목을 잰다 (log `place_test.log`)

## 10/15 16시 — 다시 시작 (가장 최신, 이것부터)
- 사용자: "다시 계속해줘 정확도 100 나오게 모든 기술 논문 적용해서 ... 내가 그만하라 할때까지"
- al1 결과: 3cm 17.1% (혼자 26 / 바꾸기 16.7 / 돌아오기 3.3), 판 안 32.9%
- 정렬 보조 시험 (al1, 8항목 160장면): 3cm 16% → 33%, 판 안 29% → 49%. `run_alr_assist.sh` 가 alr1 + 보조 28항목도 잰다 (log `alr_assist.log`)
- 돌고 있는 것
  1. `run_assist_now.sh` (log `assist_now.log`): 정렬 보조(`--grasp-assist`) al1, T0·T5·T8·T6 + A8→B5, A4→B9
  2. `run_alr_round.sh` (log `alr_round.log`): 1 이 끝나면 alr1 학습 (al1 에서, 손 기준 물체 위치 `relfeat.py`, `--feat relfeat --reset-feat`) ‖ al1 교정 시범 2개 → alr1 28항목 → alr2 → 28항목
- 새 것: `relfeat.py` (정면 카메라 2차 변환식으로 블록·판의 '손 기준 m' 16칸 + 손목 화면 8칸, 블록은 손이 10cm 넘게 떨어졌을 때만 새로 잼, 판은 처음 값),
  `piper_sim/calib_front.py`·`front_calib.json` (판까지 8개, 평균 0.05~0.11cm), `grasp_assist.py`. 시범에서 쥐는 순간 손→목표 블록 거리 중앙 0.09cm
- 모델 폴더 `relfeat` 표시가 있으면 평가(Runner.prep)가 장면마다 기억을 비우고 손 기준 입력을 만든다

## 10/15 10시 — 오늘은 여기까지 (가장 최신, 이것부터)
- 사용자: "오늘은 여기까지하자". al1 28항목 평가(`run_al_round.sh`, log `al_round.log`)만 끝까지 두고, 끝나면 점수표 `outputs/piper/score_piper_al1h(.md, _loose.md)` 만 남는다
- 끈 것: `run_al_round2.sh` (al1 교정 시범 → al2·al3·al4), `run_assist_test.sh` (정렬 보조 시험), 매시간 점검
- 다시 시작할 때
  1. al1 점수표 확인 (bo1 평균 13.9%, 지연 9 의 bo1 T0+T8 3cm 30%·판 안 45%)
  2. `./run_assist_test.sh` 를 GPU 가 빈 상태로 (al_round.log 에 '끝' 이 있으면 5분 뒤 바로 시작) — 정렬 보조(`--grasp-assist`) 켬/끔 비교
  3. `./run_al_round2.sh` (al1 교정 시범 → al2 → 28항목 → al3 → al4). 사진 특징 계산본 `cache/vis` (3장에 1장) 그대로 씀
- al1 지금까지 (단독 5과제 100장면): 판 안 44%, 3cm 22%. 실패는 쥐었다 빠짐 30 (21건이 1스텝), 판 밖 20, 못 집음 6

## 10/14 19시 — 밤새 도는 순서 (가장 최신, 이것부터)
- 사용자: "내가 그만하라고 할때까지 모든 기술 총동원해서 정확도 성공률 100%" → 시간 제한 없이 계속
- 오늘 결론: 지연을 줄여도 bo1 은 거의 그대로(지연 0 은 못 쥠). CUDA 그래프로 실제 지연 9스텝 (평가·교정 시범 `--cuda-graph`, 출력 같음).
  실패 영상: 손목 화면에서 블록을 가운데로 못 맞춘 채 내려가 손가락이 윗면에 얹힘 → 시범을 align 방식으로
- 순서 (모두 자동):
  1. `run_align_collect.sh` (log `al_collect.log`) align 시범 ‖ `run_al_dagger.sh` (log `al_dagger.log`) bo1 교정 시범 align 으로 — 일 바꾸기·돌아오기 부분이 GPU 메모리 부족(같이 돌린 속도 시험 탓)으로 죽어 `run_al_dagger2.sh` (log `al_dagger2.log`) 로 다시
  2. `run_al_round.sh` (log `al_round.log`): 사진 특징 계산 `cache/vis` (3장에 1장, `stride.txt`) → al1 학습 → al1 28항목 (지연 9)
  3. `run_al_round2.sh` (log `al_round2.log`): al1 교정 시범 → al2 → 28항목 → al3 → al4
- 고친 것: vis_cache 가 위치 입력 모델(상태 32칸)에서 깨짐, 미리 계산본을 쓸 때 위치 입력이 빠짐(작은 학습으로 찾음), 3장에 1장 고르는 기준을 파일 이름으로
- 감시: `watch_blk.sh`, `mem_guard_piper.sh` 다시 켬

## 10/14 13시 — 다시 시작
- 사용자: "act 다해서 한번 계속 해볼까 정확도 100 으로 해보자" — ACT 논문(08_개인논문_ACT적재)이 끝나 SmolVLA·PiPER 블록 작업 재개. 목표 정확도 100 근접
- 방향: 가장 큰 원인을 지연(0.56초 = 11스텝)으로 보고 ① 지연을 줄이기 ② 지연에 덜 민감한 시범으로 바꾸기
  - `run_lat_probe.sh` (로그 `outputs/piper/lat_probe.log`): bo1 지연 0·3·6 스텝 T0·T8 + 3.5cm 블록(지연 11)
  - `fastinfer.py`: 같은 계산을 CUDA 그래프로 (평가 `--cuda-graph`). `python fastinfer.py --policy ...` 로 원래와 출력 비교·시간 (GPU 혼자일 때)
  - 시범 새 방식 `PIPER_GRASP_STYLE=align` (task_experts): 블록 위에서 먼저 맞추고 똑바로, 마지막 3cm 천천히, 3~6스텝 멈췄다 쥠 (ACT 의 '여유 두고 잡기')

## 10/7 17시 — 잠시 멈춤
- 사용자: "논문 act로 할거라 잠시 쉬자" → SmolVLA·PiPER 블록 작업을 모두 멈췄다. 다시 하라는 말이 있기 전에는 띄우지 않는다
- 멈춘 것 (모두 PID 로 종료): `run_obj_round.sh` 와 bo2 학습(16:32 시작, 13분째 — 저장본 없음), `run_c35_probe.sh` (GPU 대기 중이었음), 작은 블록 시범 점검, `watch_blk.sh`, `mem_guard_piper.sh`, 매시간 점검 cron
- 마지막 성적: bo1 (`outputs/piper_bo1_model/merged`) 3cm 기준 단독 19.5%·전환 12.5%·재개 7.5% (평균 13.9%), 판 안 34/22.5/11.7%. ba1 과 차이 없음
- 진단: 쥐는 비율 80% 인데 쥔 것 중 판 안 42%. 실패는 쥐고 6스텝 만에 빠짐 — 손가락이 오므리는 방향으로 1.4~1.5cm 어긋남(4.5cm 블록 여유 1.25cm), 손 높이 +3.5cm (4장면)
- 다시 시작할 때
  - bo2: `data/piper_blk_dagger_bo1_t` (427개, 검출 계산까지 끝남) → `run_obj_round.sh` 의 [4] 줄(bo2 학습)부터
  - 작은 블록: `PIPER_CUBE=0.035` (블록 3.5cm, 막대 9x3.5x3.5, 폴더 `assets/blocks_c35`·`bddl_blocks_c35`). 시범 프로그램 점검 T0·T1·T4·T5·T6·T8·T9·전환 3쌍·재개 2쌍 모두 5/5, T2·T3·T7 은 막대 굵기 고친 뒤 다시 재야 함. 시험 `run_c35_probe.sh`, 수집 `run_c35_collect.sh` (아직 안 돌림)
  - 평가 기록에 `hold_segs` ([시작, 끝, 물체, 끝날 때 그리퍼 명령, 오므리는 방향 어긋남cm, 바닥 위 높이cm]), `objects_final`, `eef_final` 이 새로 남는다 (switch_experiment.py)

## 10/4 색 블록 장면

### 10/6 낮 (14시) — 이것부터 (가장 최신)
- 사용자 결정: **VLA + 색·모양 검출**. `objdet.py` 가 정면·손목 사진에서 블록 4·판 4 의 화면 위치 24개를 찾아 로봇 상태 8개 뒤에 붙인다 (max_state_dim 32)
- 학습: `train_lora.py --objfeat` (에피소드 옆 `*.objfeat.npy` 필요 — `objfeat_cache.py`, `run_objfeat_cache.sh`). 처음 붙일 때 state_proj 새 열 0, 정규화 통계 32칸. 모델 폴더에 `objfeat` 표시 → 평가(`Runner.prep`)가 자동으로 검출해 붙임
- 돌고 있는 것: ba2 학습(절대 목표, 넓은 배치·회복·교정 시범, 로그 `abs_round2.log`, round2 스크립트는 멈춤) / 검출 미리 계산 (`objfeat_cache.log`) / `run_obj_round.sh` (로그 `obj_round.log`: 검출 끝 → bo1 학습 → ba2 일부 항목 → bo1 28항목)
- ba1 28항목: 단독 19%·전환 14%·재개 12% (3cm), 판 안 31/24/21%
- 카메라 위치 그림: `LAB-1309/05_로봇팔_제원/img/piper_cameras.png`

### 10/5 밤 (21시) — 이것부터 (위 10/4 밤 절보다 최신)
- b1·b1c·A2C2·지연 줄이기 모두 T8 15~20%에서 막힘. 원인: **변화량 동작의 오차 누적** (시범을 눈 감고 다시 실행: 변화량 8/12, 절대 목표 12/12)
- 새 방식: `train_lora.py --abs-pos --renorm-action` (위치 3개 = 그때 손 위치 + 5cm×명령, 모델 폴더에 `abs_pos` 표시) → 평가는 `switch_experiment.Episode.to_delta` 가 자동 변환. A2C2 는 변화량 모델용이라 절대 목표 모델에 쓰지 말 것
- 성공 기준 두 가지 기록 (사용자 결정): `a_success`/`b_success`/`a_resume_genuine` (LIBERO 3cm), `*_loose` (판 안). 성적표 `scoreboard.py --blocks [--loose]`, 분모 = 전체 시도
- 쥠 판정: PiPER 는 두 손가락이 모두 물체에 닿아야 (`sx.HOLD_CONTACT`)
- 돌고 있는 것: `run_abs_round.sh` (로그 `outputs/piper/abs_round.log`): b2 학습 끝 → ba1 학습(b2 에서) ‖ b2 28항목 → ba1 28항목
- 감시: `watch_blk.sh` (2분마다 `outputs/piper/STATUS.txt`, 멈춤 `watchdog.log`)
- 다음: ba1 이 좋으면 교정 시범(ba1) → ba2 → 1000~1099 최종 (`run_piper_final.sh`, 블록은 `PIPER_BLOCKS=1`)

### 10/4 밤 (23시) 지금 도는 것 — 이것부터
- b1 0% 원인: ① 평가 지연 흉내가 시작할 때 첫 계획 앞 11개를 버림(고침, `policy_action_async` 의 k) ② 96~97cm 에서 오므림(시간으로 오므리는 버릇 가설 → 시범 속도·멈춤 다양화) ③ 지연 때 2~3cm 빗나감
- 평가 규칙: PiPER 는 빈손(손가락 0.000)이면 쥔 것으로 안 봄 (`sx.GRIPPER_MIN_QPOS=0.005`)
- 돌고 있는 것: b1c 학습 (`outputs/piper_b1c_model`, 사진 흔들기 끔, 블록 시범 2배, 24000스텝, 로그 `outputs/piper/train_b1c.log`), b1 교정 시범 (`data/piper_blk_dagger_b1`, 로그 `dagger_b1_*.log`), 3차 수집 (`run_blk_collect3.sh`, 속도·멈춤 다양화)
- 자동 순서: `run_blk_round_d.sh` (로그 `blk_round_d.log`: b1·b1c 비교 → b2 학습 ‖ b1c 28항목 → b2 28항목 ‖ A2C2 → b2+A2C2) → `run_blk_round_e.sh` (로그 `blk_round_e.log`: b2 교정 시범 ‖ 반복 5단계 시험 → b3 → 28항목 → A2C2)
- 멈춘 것: b0n(비교용, 학습법이 바뀌어 나중에 같은 방법으로 다시), run_blk_round1/2/c (D·E 가 대신)
- 최종: `PIPER_BLOCKS=1 ./run_piper_final.sh outputs/piper_b3_model/merged blk_final 100 outputs/a2c2_piper_b3` (반복 5단계가 낫다면 `NS=5 LAT=8`)

- 사용자: 물체를 색 블록·색 판만으로("그냥 사각형만 있는 걸로"), 목표 "성공률 정확도 100%", 논문은 **이번 대회 시뮬레이션만, 다음 대회 실물**
- 장면: `piper_sim/blocks.py` (`PIPER_BLOCKS=1` 환경 변수로 켬 → `piper_robot.use_piper_in_lerobot()` 가 `pb.install()`, BDDL 은 `piper_sim/bddl_blocks`)
  - 블록 4 (빨강=akita_black_bowl_1, 초록=cream_cheese_1, 파랑=blue_block_1, 노란 막대=wine_bottle_1), 판 4 (보라=plate_1, 회색, 주황, 흰색). 과제 10개 `pb.TASKS`
  - 시범 프로그램: 모든 과제 `te.block_task` (grasp_bowl_safe → block_grasp_candidates 4방향 × 기울기 6, carry_place z_safe = 가장 높은 블록 + 2.25 + 3cm), `te.block_keepout`
- 10/4 고친 것: 성공 판정이 예전 BDDL 을 읽던 것(`sx.get_libero_path` 교체), PiPER 손 닫힘 기준 0.030(`sx.GRIPPER_OPEN_QPOS`, 0.035 에서 깜박), 쌓은 블록 피하기, 기울기마다 따로 고르기, 막대 180° 방향
- 시범 프로그램 점검 (학습 장면 2000~2019): 혼자 20/20 (2014 고친 뒤), 일 바꾸기 12쌍·돌아오기 6개 모두 10/10 (`outputs/piper/blk6_s*.log`). 새 배치 점검 `outputs/piper/blk_held_t.log`
- 자동 순서 (setsid):
  - 정상 수집 `data/piper_blk_normal` (12:00 시작, 10과제 × 2000~2099, 로그 `blk_collect_n{1,2}.log`) + `run_blk_collect.sh` (전환·재개 12쌍 × 2000~2049 → `data/piper_blk_switch`, 끝나면 `_t` 로 잘라 냄, `outputs/piper/blk_collect.log` 에 "블록 수집 끝")
  - `run_blk_round1.sh` (로그 `outputs/piper/blk_round1.log`): 수집 끝을 기다림 → **원래 물체 piper_s2 학습을 멈추고** 그 마지막 저장본에서 `outputs/piper_b1_model` 학습(20000스텝) → 쉬운 점검 → 새 배치 1000~1019 28항목 (`outputs/piper_b1h_*`) → A2C2 → `piper_b1ah_*`. b1 학습이 끝나면 논문 비교용 `piper_b0n`(정상 시범만, 같은 출발·스텝)을 평가와 나란히 학습 → `piper_b0nh_*`. 수집 가속: `run_blk_collect_boost.sh` (정상 수집이 끝나면 전환 수집기 2개 추가, 2025~2049)
  - 메모리 감시 `mem_guard_piper.sh` (collect_piper.py 만, 4GB 아래면 최근 수집기 멈춤 — 다시 돌리면 이어서)
- 멈춘 것: `ab_numsteps.sh` — "지연 없음"을 앞부분 없이 재서 0/10 (Panda 와 같은 잘못된 조건). 블록 모델에서 반복 단계(num_steps 5)만 다시
- 다음: b1 성적 → 95% 미만 항목 블록 DAgger(새 데이터 이름 `data/piper_blk_dagger_*`, run_piper_round.sh 는 원래 물체용 데이터 목록을 쓰니 고쳐서) → 1000~1099 최종


## 10/3 새 방향: WeGo Robotics 가 공급한 AgileX PiPER 로 95% (사용자 지시, 가장 우선)

- Panda(LIBERO) 모델은 **기록용으로 끝까지 만든다** (v6c2 → run_post → v6d → v6e, 아래 절). PiPER 가 본 실험
- PiPER 시뮬 파일: `piper_sim/` — `build_piper_xml.py`(로봇·그리퍼 XML), `piper_robot.py`(robosuite 등록, `use_piper_in_lerobot()`),
  `layout.json`(밑동 (-0.38,0,1.19), 29cm 받침대), `init_qpos.json`(처음 자세), `make_bddl.py`(캐비닛을 로봇 쪽 −x 로 돌린 LIBERO-Goal 변형 → `piper_sim/bddl`),
  `ik.py`(역기구학으로 관절 길을 미리 계산해 따라가기 `servo_path`), `audit_piper.py`(시범 프로그램 성공률)
- 점검: `python diag_teacher.py --piper --task 0 --episodes 2000 2001 --gif` / `python piper_sim/audit_piper.py --tasks ... --episodes 2000-2009 --out ...`
- 현재 (10/3 09시):
  - 장면: 캐비닛 (0.06~0.08, −0.24~−0.22) yaw −π/2 (서랍이 −x), 선반 (−0.37~−0.35, −0.43~−0.41) yaw 0 (홈이 로봇 쪽으로 올라감)
    `make_bddl.py --cab-yaw -1.5707963 --cab 0.06 -0.24 0.08 -0.22 --rack -0.37 -0.43 -0.35 -0.41 --rack-yaw 0`
  - 제어: `piper_sim/ik_controller.py` (손끝 목표 → 역기구학 → 관절 PD). `piper_robot.USE_IK_CONTROLLER=True`. 관절 마찰 0.05
  - 시범 프로그램 (모두 `task_experts.EXPERT`, PiPER 면 PiPER 판으로 갈라짐): T0·T3 `open_drawer_grasp`(45°, IK 가지 고르기, 위 서랍 반쯤 오므림),
    T3 `drawer_bowl_piper`(준비 자세, 피할 상자), T2·T9 `wine_grasp_piper`/`wine_place_piper`(60° 숙여 쥠, 높이 지나가기 `pik.transit`),
    T7 `turn_on_stove_piper`(15° 기울임), 그릇 옮기기 `se.carry_place` PiPER 판(그릇 축 둘레 손 회전 후보 + transit), `_grasp_bowl_piper`(기울인 후보, 특이 자세 피함)
  - 개발 장면 결과: 10개 모두 성공 (T0·T1·T3·T5·T6·T7·T9 10/10, T2·T4·T8 고친 뒤 실패 장면 모두 성공)
  - 도는 것: 새 배치 1000~1019 단독 점검 `outputs/piper/audit_h_{a,b,c}.log`, 전환 12쌍 `outputs/piper/audit_sw_dev.log`
  - 수집 도구: `piper_sim/collect_piper.py normal|switch ...` (collect_v6 와 같음, 2000번대만)
- 10/3 23:30: **PiPER 1차(piper_s1) 쉬운 점검 0/5** — 그릇 테두리를 0.7cm 높게(95.6cm) 닫아 빈손. 시범이 테두리 끝 0.2cm 만 걸쳐 쥐었다(94.9cm, 테두리 위 95.1cm)
  → `task_experts.PIPER_GRASP_DEEPER=0.012`(1.2cm 깊게) + 마지막 접근 tol 0.004. 1.2cm 높게 닫아도 6/6 잡힘
  → `run_piper_collect3.sh`: 그릇 시범만 다시(정상 T1·T3·T4·T8 × 2000~2099, 전환·재개 12쌍 × 2000~2049) → `data/piper_keep_t`(그릇 없는 예전 시범) + `piper_normal3_t` + `piper_switch3_t` → `outputs/piper/train_data.txt` → 라운드 s2 학습 데이터
  → s1 새 배치 28항목 평가는 계속 (기록용), 라운드 s2 DAgger 는 수집이 끝난 뒤 새 코드로
- 10/3 21시: Panda v6c3 기록 성적표 끝 (85% 이상 1/28, `06_결과물/Panda`). PiPER 2차 수집 끝(정상 398, 전환 881). 고친 것: `se.secure_hold` PiPER 기준 3cm(잡음 A1→B8 실패), `switch_experiment` PiPER 평가 horizon 4000(600스텝 제한에서 평가가 죽던 것). 평가·DAgger 경로를 v6b 로 미리 시험함
- 10/3 16:20: PiPER 평가 시간 제한 600스텝 (`piper_sim/eval_piper.py` 가 자동으로 넣음). Panda 기록 평가는 PiPER 학습 중에만 (`swap_panda_record.sh` → `run_panda_record2.sh`, GPU 11GB 넘침 방지)
- **10/3 14:45 사용자: "panda는 끝나는대로 바로 모델 만들어줘 / 이제 piper만 해야해"** → Panda 는 v6c3 기록 평가만(`run_panda_record.sh`, 2프로세스, `outputs/panda_v6c3_rec_*`), 개선은 PiPER 만
- 10/3 16시 자동 순서:
  - `run_piper_train.sh` (15:15 시작): PiPER 1차 `outputs/piper_s1_model` (v6b 에서, `data/piper_*_t` = 오래 멈춘 구간 잘라 낸 시범) → 쉬운 점검(지연 11) → 새 배치 20장면 28항목 → `outputs/piper/score_piper_s1.md`
  - `run_piper_collect2.sh`: 고친 시범 프로그램으로 2차 수집 `data/piper_normal2`(2060~2099), `data/piper_switch2`(2030~2059) → `_t` 잘라 냄
  - `run_piper_chain.sh` → `run_piper_round.sh piper_s1 piper_s2 2100-2129` → `... piper_s2 piper_s3 2130-2159` (95% 미만 DAgger → 학습 → 평가 → A2C2 → 평가). 로그 `outputs/piper/round_s*.log`
  - Panda v6c3: 쉬운 점검(지연 없음) 0/5 지만 배치 조건(지연 11 + ttrtc) 2/5 → 정규화 가설 기각. 지연 흉내내기 모델은 반드시 배치 조건으로 잴 것
- PiPER 시범 프로그램 (10/3 16시, 새 배치): 단독 10개 각 20/20, 전환 12쌍·재개 6개 10/10 (A4→B1·A8→B3 고친 뒤 실패 장면 성공)
  - 고친 것: 멈춤 끊기(`se.STALL_STEPS`), 놓을 자리까지 보는 쥐기(`_grasp_bowl_piper(dest)`), 팔 링크 충돌 검사(`ArmIK.arm_hits`), 처음 자세 자동 우회(`ik._home_via`), 선반 뒤 준비 자세(`post_place_ready`), 캐비닛·선반 피할 상자(`ik.scene_keepout`)
- (이전) 10/3 10시 자동 순서 (setsid):
  - `run_piper_collect.sh` → `data/piper_normal`(10과제×2000~2059), `data/piper_switch`(12쌍×2000~2029), 로그 `outputs/piper/collect_*.log`, 끝나면 run_collect.log 에 "PiPER 수집 끝"
  - `run_piper_train.sh` → 수집 끝 + v6c3 학습 끝 + v6c3 쉬운 점검 결과를 기다린 뒤 PiPER 1차 학습(`outputs/piper_s1_model`, v6b 에서) → 쉬운 점검 → 새 배치 20장면 28항목 (`outputs/piper_s1h_*`, 성적표 `outputs/piper/score_piper_s1.md`)
  - 학습 중 `outputs/piper/TRAINING` 파일이 있으면 Panda run_post 의 무거운 평가가 기다린다. `outputs/v6/STOP_ROUNDS` 가 있으면 Panda 라운드(v6d·v6e)는 시작 안 함
  - PiPER 시범 프로그램 새 배치 1000~1019: 10과제 모두 20/20 (200/200). 전환 점검 새 배치 `outputs/piper/audit_sw_h.log`
- Panda: v6c2 쉬운 점검 0/5 (정규화 다시 재기 의심) → `run_v6c3.sh` (정규화 그대로, 07:48 시작, 로그 `outputs/v6/run_v6c3.log`) → run_post → v6d → v6e
- 다음: 점검 결과 보고 → 95% 미만 고치기 → PiPER 수집(정상 60×10, 전환 30×12) → PiPER 학생 학습 → A2C2·DAgger → 1000~1099 평가

# (10/2 16:40 기준 Panda 기록)

## 지금 목표 (사용자 지시)

- 모든 항목 각각 **95% 이상** (10/2 저녁 85%→95% 상향; 단독 10개, 전환 12쌍 B 성공, 재개 6쌍), **10/10까지**, 모든 기술, 토큰 다 써도 됨
- 최종 판정 (10/2 밤, 사용자 "장면 최대한 많이"): 항목마다 **새 배치 100장면(1000~1099)에서 95번 이상**. `run_final_eval.sh POLICY OUT 100 [A2C2DIR] [DIMS] [NA]`. 시간 남으면 200장면. 10번 평가는 중간 점검용
  (참고: 30장면 29번은 하한 83%, 100장면 96번은 하한 90%)
- 선생(스크립트 전문가)이 95% 못 넘는 항목은 학생도 어렵다 → `run_teacher_audit.sh` (v6c 학습 끝나면 자동, 결과 `outputs/teacher_audit/*.json`, 실패 이유 포함). 95% 미만 항목은 전문가부터 고칠 것
- **실제 로봇에 쓸 수 있어야 한다**: 카메라 2대 + 로봇 상태 + 지시문만. 평가는 **추론 지연 11스텝**(1080 Ti 0.56초)을 넣고 한다
- 10/10까지는 근무 시간 규칙(평일 9~18시)보다 이 지시가 우선
- ⚠️ 메모리 31GB, 스왑 없음. 사용자가 RustDesk로 원격 접속 중 — 넘치면 멈춘다. 학습(18GB)은 다른 무거운 일과 겹치지 않게

## 지금 돌아가는 것 (10/2 11:55 시작)

`run_dagger_v6b.sh` (setsid) — 로그 `outputs/v6/dagger/p{1,2,3}.log`, `outputs/v6/v6blat.log`
- DAgger 3개: v6b 모델을 지연 11스텝으로 돌리다가 무작위 시점에 선생 스크립트가 넘겨받음 → `data/v6_dagger_lat/episodes`
- v6b 지연 평가 (장면 20~29, 10회): `outputs/v6blat_switch`, `outputs/v6blat_forget`
  - 첫 결과: A8→B0 0%, A8→B3 0% — v6b는 지연에 무너진다

## 새로 만든 것 (10/2, 논문 조사 → LAB-1309/02_논문노트/지연과_정밀도_논문조사.md)

| 파일 | 내용 | 검증 |
|---|---|---|
| `ttrtc.py` | 학습 때 지연 흉내내기 (arXiv 2512.05964). 묶음 앞 d스텝 정답 고정, 뒤만 학습. 추론 때 기다리는 동안 할 동작을 앞에 넣음 | 끈 상태에서 원본과 손실·출력 **완전히 같음**, 앞부분 고정 확인 |
| `train_lora.py --rtc-max-delay 14 --ema 0.999` | 위 + 가중치 이동평균 | CPU 60스텝 점검 중 |
| `switch_experiment.py --ttrtc` | 지연 모사 중 계산 시작 때 남은 계획 앞 L개를 고정 (flush 직후처럼 계획이 비면 넣지 않음) | 컴파일 |
| `a2c2.py gen/train`, `switch_experiment.py --a2c2 DIR` | 매 스텝 보정 네트워크 (arXiv 2509.23224). ResNet18(GroupNorm)×2카메라 + 상태 + 묶음 동작 + 위치 k → 보정값 | 작성만 |

## 10/2 13시 발견: 동작 정규화가 선생 데이터와 안 맞았다

모델은 동작을 (값 − 평균) / 표준편차로 바꿔 배운다. 그 평균·표준편차가 **LIBERO 사람 시범 기준**이었다.
선생 데이터로 재 보면 위치 동작은 0.6~0.7배로 작고, 손목 회전은 1.2~2.0배(최대 12.8배)로 크다.
그래서 학습 손실이 손목 회전에 쏠리고 **손 위치(그릇 잡는 자리)의 정밀도를 덜 본다.**
→ `train_lora.py --renorm-action` 추가(학습 데이터에서 다시 잼, 전·후처리 둘 다 바꾸고 저장). 저장 후 다시 읽기까지 오차 0 확인.
→ v6c 학습에 넣음.

참고: 원래 SmolVLA(사람 시범 50개로 학습)는 지연 없이 T1 100%, T3 80%, T4 90%인데, v6b는 63%, 3%, 50%. 선생 방식이 배우기 어렵거나 정규화 문제일 수 있다.

## 지금 도는 자동 순서 (setsid)

- `run_dagger_v6b.sh`: DAgger p1·p2·p3 + v6b 지연 평가
- `dagger_split.sh`: p1·p2가 끝나면 p3의 남은 쌍 절반을 p4가 뒤에서부터 맡는다 (로그 `outputs/v6/dagger/split.log`)
- `run_v6c.sh`: DAgger가 모두 끝나면 v6c 학습(지연 흉내 + EMA + 정규화 다시 잼) → A2C2 데이터 + 빠른 비교 → A2C2 학습 + 정식 평가 → v6c+A2C2 평가 (로그 `outputs/v6/run_v6c.log`)
- `make_gif_policy.py`: 평가와 똑같이 돌려 GIF. ⚠️ `CUDA_VISIBLE_DEVICES=""`로 돌리면 MuJoCo EGL이 죽는다 (GPU 필요)

## 10/2 13시 측정: 학습 속도의 한계

- 1080 Ti 학습 한 스텝(4장) 0.87초(혼자일 때). 그중 **사진 처리(SigLIP) 60%**, 나머지 40%.
- 사진 특징 미리 계산(`vis_cache.py`, `train_lora.py --vis-cache`)은 손실 차이 0.2%로 정확하지만, 만드는 데
  초당 약 8프레임 → 75만 프레임에 **27시간**. 데이터를 1번도 다 못 도는 지금은 손해 → **쓰지 않음**(코드는 남김).
- 그래서 정밀도는 빠르게 학습되는 A2C2(ResNet18 128px)에 맡긴다. `run_v6c.sh` [0]에서 v6b 위에 작게 먼저 시험.
- ⚠️ GPU 4개 + 시험 스크립트를 같이 돌리다 v6b 지연 평가 A1→B7이 메모리 부족으로 죽었다 → 따로 다시 돌림
  (`outputs/v6/v6blat_A1B7_rerun.log`). **GPU 프로세스는 4개를 넘기지 말 것.**

## ⚠️ 10/3 0시 — v6c 폐기, v6c2 로 다시

- v6c 는 그릇을 집자마자 놓는 버릇 (T8 학습 배치·지연 없음 0/5, v6b 3/5). 원인: 낮 DAgger 시범 267개가 쥔 물체를 바로 놓고 다시 집음
  → `data/v6_dagger_lat_quarantine` 로 격리, `dagger_v6.takeover` 수정(쥔 채 이어서 / 살며시 내려놓기), 저장 전 검사 추가
- 지금: `run_v6c2.sh` — v7_normal 수집 끝나면 v6c2 학습(`--frame-stride 2` 로 메모리 절반) → `run_post.sh v6c2` (쉬운 점검 2/5 미만이면 멈춤, 새 배치 평가, A2C2) → `run_round.sh` v6d, v6e
- v6c 결과(`outputs/v6ch_*`)와 `data/a2c2_v6c` 는 쓰지 말 것

## 10/2 21시 이후 추가

- 장면 번호: 0~19(와 %50 이 0~19) 학습 고정 배치 / 1000~1999 평가 전용 무작위 / **2000 이상 학습 전용 무작위**
- `run_v7_collect.sh`: v6c 학습 끝나면 고친 시범 프로그램으로 2000번대에서 정상 600·전환 480 수집 → `data/v7_*` (run_round 학습 데이터에 자동 포함)
- 시범 프로그램: 처음 자세 피해 가기 (`scripted_expert.HOME_AVOID=0.12`, `_home_detour`, `_keep_off_home`) — 확인 중 `outputs/teacher_audit/detour2.json`
- 나중에: `run_teacher_heldout.sh` (시범 프로그램 100장면 새 배치 측정) — 수집이 끝난 뒤 띄울 것 (메모리)

## ⚠️ 10/2 21시 — 평가 장면 누수 발견과 수정 (가장 중요)

- LIBERO 고정 시작 장면은 `init_states[번호 % 50]`. 학습 수집에 장면 50~139 를 써서 **평가 장면 20~49 배치가 학습 데이터에 들어 있었다** (장면 20=70=120 배치 동일 확인)
- 수정: `switch_experiment.HELDOUT_FROM = 1000` — 1000 이상은 고정 장면 대신 무작위 배치(번호=시드, 재현됨). **학생 모델 평가는 이제 장면 1000번대만.** `collect_v6.ep_range` 는 %50 이 20~49 인 번호와 1000 이상을 막는다
- 그 전의 우리 모델(v5·v6·v6b·A2C2) 평가 숫자는 누수 조건. 새 배치에서 다시 잼 (`outputs/v6bh_*`, `v6ch_*`, `v6cah_*`)
- 지금 도는 것: `run_v6c_post.sh` (v6c 학습 끝나면 새 배치 평가·A2C2) → `run_chain.sh` (v6d, v6e 라운드, 평가 새 배치)
- 시범 프로그램 수정: 접시 밀기 `push_plate_v3`(T5·A4→B5·A8→B5 30/30), 서랍+그릇 `drawer_bowl(pre_clear=0.128)`(T3 실패 3장면 → 성공, 30장면 확인 중 `outputs/teacher_audit/round2.json`)

## 10/2 18시 — 자동 라운드 (밤새)

- `run_chain.sh`: `run_v6c.sh` 가 끝나면 `run_round.sh v6c v6d ...` → 이어서 `run_round.sh v6d v6e ...`
- `run_round.sh PREV NEXT ...`: `plan_round.py` 가 결과로 설정을 정함(n_act, A2C2 사용·차원, 95% 미만 항목)
  → 그 항목만 DAgger(실제 배치 설정: 지연 11, `--ttrtc`, `--strategy keep`, A2C2) → NEXT 학습(`--dagger-frac 0.3`, 통계는 PREV 것)
  → A2C2 데이터·학습 → NEXT, NEXT+A2C2 28개 항목 평가 → `outputs/v6/score_*.md`, 로그 `outputs/v6/round_*.log`
- 선생 점검: `outputs/teacher_audit/all.json` (장면 20~49, 노이즈 없음). A8→B7 은 0~19 에서 B 20/20, 재개 20/20 (수집 데이터의 68% 는 노이즈 탓)

## 10/2 16:40 상태 — 검증된 결과

| 무엇 | 결과 (실제 로봇 조건: 지연 11스텝, 장면 20~29, 10회) |
|---|---|
| v6b 28개 항목 | 85% 이상 **0/28**. 최고 A8→B7 70%. T2는 지연 없이 100% → 지연 0% (맴돌기) |
| v6b + A2C2 (15% 데이터, 8000스텝) | T1 3→**10**/10, T4 5→**9**/10, T8 0→**6**/10, A8→B5 6→7, A8→B7 7→**3** (떨어짐) |
| A8→B7 떨어진 이유 (가설) | 실패 때 손은 손잡이까지 가는데 그리퍼가 열린 채 끝남 → 그리퍼 보정이 방해? `run_a2c2_dimtest.sh` 가 v6c 학습 끝나면 `--a2c2-dims posrot` 로 확인 (결과 `outputs/v6ba_posrot`) |
| DAgger | B 구간 651개 + 재개 구간 119개 (`data/v6_dagger_lat`) |

지금: `run_v6c.sh` [1] v6c 학습 16:00 시작 → 약 22:00 끝 → [2] A2C2 데이터 + 빠른 비교 → [3] A2C2 학습 + v6c 정식 평가 → [4] v6c+A2C2 평가.
결과 보는 법: `python scoreboard.py --forget outputs/v6clat_forget --switch outputs/v6clat_switch`, A2C2 는 `outputs/v6ca_*`.

다음 (사람이 결정): 85% 못 넘은 항목만 DAgger 2차 (v6c + 효과 있으면 A2C2, `dagger_v6.py --ttrtc --strategy keep --n-action-steps NA --a2c2 DIR`, 재개는 `--resume`)
→ v6d 학습은 v6c 에서 이어서 (**`--renorm-action` 다시 넣지 말 것** — v6c 에 저장된 통계를 그대로 써야 한다)

## 다음 순서

1. DAgger 끝나면 v6c 학습 (v6b에서 이어서, normal + switch + dagger_lat, `--full-expert --aug --balance --workers 0 --rtc-max-delay 14 --ema 0.999`, 약 20000스텝, 5~6시간)
2. v6c 지연 평가: `--latency-steps 11 --ttrtc`, n_action_steps 1(계속 계산) vs 10, 전환은 keep(멈춤 없음) vs flush
3. A2C2: `a2c2.py gen --policy outputs/v6c_model/merged --ttrtc` → `a2c2.py train` → `--a2c2` 로 평가
4. DAgger 2차 (v6c + A2C2) → v6d ...

## 경쟁 논문

Causeway (arXiv 2609.30913, 9/25): LIBERO-Goal 전환, 재개 없음, 추론 2.5배 느림 → `LAB-1309/02_논문노트/Causeway.md`

---

# (옛) 인계 메모 (2026-09-21 11:40 기준)

> 세션이 끊기면 이 파일부터 읽으세요. 연구 맥락은 `LAB-1309/CLAUDE.md`,
> 결과는 `LAB-1309/04_실습/05_libero/README.md`, 오늘 일지는 `LAB-1309/03_일지/2026-09-21.md`.

## 0. 근무 시간 (2026-09-18 지시)

**한국시간 평일 09:00~18:00 에만 작업.** 주말·공휴일과 18시~다음날 9시는 쉰다.
- 추석 연휴 **9/24(목)~9/26(토)**, 한글날 **10/9(금)**, 개천절 대체 **10/5(월)** 추정
- **이미 돌고 있는 GPU 큐는 밤새 계속 돌게 둔다** (연구실 PC 의 일). 아침에 결과를 분석한다
- 세션 시작 시 `TZ=Asia/Seoul date` 로 확인

## ⛔ 제약 (절대)

- 전환할 때 **초기 자세로 복귀 금지** (캘리브레이션/리셋이 됨)
- **사람처럼 자연스럽게** 이어져야 함 (stop-and-go 금지)
- 물체를 아무 데나 내려놓는 것은 허용
- `retreat`/`ret_pos`/`ret_rot` 은 **진단·상한 전용**. 증류 대상이 될 수 없음

## 1. ⭐ 지금 연구가 어디에 서 있는가

### 진짜 원인을 찾았다 (9/21)

| 측정 지점 | CMI (지시문이 동작을 가르는 정도) |
|---|---|
| 에피소드 시작 | 0.719 |
| **물체를 쥔 직후** | **0.284 (−61%)** |
| 들고 이동 중 | 0.761 (회복) |

**물체를 쥐는 순간 정책이 지시문에 귀를 닫는다.** 태스크 1·2·4·8 모두 같다.
→ 지금까지 실패한 대책들은 전부 **"몸"(자세·노이즈)** 을 고치려 했다. 문제는 **"귀"** 였다.

### 닫힌 길 (다시 열지 말 것)

| 방향 | 최종 결과 |
|---|---|
| 노이즈 — 좋은 시드 고정 | ❌ 처음 보는 쌍 **7%** (기준 31%, p=0.031) |
| 노이즈 — CEM 으로 μ 학습 | ❌ 검증 50회 **−10%p** (p=0.885) |
| 노이즈 — 크기 조절 σ 0.5~2.0 | ❌ σ=2.0 에서 전부 0% |
| 노이즈 — 판정기로 후보 선택 | ❌ Q−V = 0.0003 |
| 학습 — LoRA 1·2·3차 | ❌ 32 / 32 / 31 / **30%** (원본 32%) |
| 회복 — 흔들기·rollback·finish_a | ❌ 전부 기준선 이하 |
| 회복 — 초기 위치로 되돌리기 | ⚠️ 통하지만(89%) **제약 위반** |

### 열린 길 (지금 여기)

**지시문 증폭 (instruction CFG)** — `v = v_B + (w−1)(v_B − v_A)`
이전 지시 A 를 음의 조건으로 쓴다. 자세를 안 건드려 제약을 지킨다.
- 구현: `instr_cfg.py`, 검증: `test_cfg_identity.py` (w=1 에서 원본과 **비트 단위 동일** 확인)
- ⏳ **지금 평가 중** (아래 2절)

## 2. 지금 돌아가는 것

```
run_cfg.sh (2622560)  — 로그 outputs/cfg.log
  w = 1.5 / 2.0 / 3.0  ×  4쌍(8→7, 4→7, 1→5, 2→5)  ×  10회 = 120 에피소드
  11:30 시작, 조건당 약 10분 → 13:30 경 종료 예상
```

기준선(w=1)은 **다시 돌리지 않는다** — w=1 이 원본과 비트 단위로 같으므로
`outputs/switch` 의 기존 결과가 곧 기준선이다.

## 3. 결과가 나오면

```bash
cd ~/smolVLA
.venv/bin/python analyze_cfg.py            # 성공률 + jerk + 실패유형 + 짝비교
```

**⚠️ 성공률만 보고 판단하지 말 것.** `analyze_cfg.py` 가 jerk 를 같이 찍는다.
증폭은 동작을 키우므로(정규화 크기 0.69 → 0.89(w=1.5) → 1.11(w=2)),
성공률이 올라도 jerk 가 같이 오르면 **"사람처럼 자연스럽게" 제약을 깬다.**

| 결과 | 다음 |
|---|---|
| 성공 ↑, jerk 유지 | ✅ 들고 이동 중 시점 + 더 많은 쌍으로 확대, 자연스러움 정밀 측정 |
| 성공 ↑, jerk ↑↑ | ⚠️ w 를 낮춰 절충점 찾기. 자연스러움 지표를 논문에 같이 보고 |
| 성공 그대로 | ❌ "지시문 채널이 원인" 가설 재검토 → 아래 4절 |

## 4. 증폭이 안 통하면 (다음 후보)

1. **지시문 토큰 반복** — 같은 지시를 여러 번 넣어 주의를 강제로 끈다
2. **주의 가중치 직접 조정** — 언어 토큰에 가는 attention 을 추론 때 키운다
3. **CMI 를 목적함수로 학습** — 기존 LoRA 는 동작을 모방했다. CMI 자체를 올리는 학습은 안 해봤다
4. **전환 시점을 늦추기** — CMI 가 회복되는 "들고 이동 중"까지 기다렸다 전환
   (즉시성을 포기하는 것이라 문제 정의가 바뀐다. 기준선으로만 의미)

## 5. ⚠️ 증폭의 알려진 약점 (효과가 확인되면 바로 고칠 것)

**추론이 2배다.** 속도장을 두 번 구한다.
1080 Ti 단독 추론이 이미 **560ms = 11.2스텝**이고,
지연 실험에서 **0.5초 지연이면 90%→12%** 였다. 그대로면 실기에서 무너진다.

→ 해결책: 두 조건을 **배치 2로 한 번에** 넣는다. 작은 모델은 배치 2가 배치 1과
시간이 거의 같아 추가 비용이 사라진다. 지금 `instr_cfg.py` 는 캐시를 **순차로**
만들고 있어 그 최적화가 안 들어가 있다. (효과 없는 방법을 최적화할 이유가 없어 미뤘다)

## 6. 미해결로 남긴 것

| 질문 | 상태 |
|---|---|
| 회전 30도 단독은 0/20 인데 **20cm+30도는 43%** — 왜? | 가설(이동 중에 손목이 고쳐진다) 세웠으나 **검증 실패**. 올바른 집단(교란된 시작)의 궤적이 필요. `pose_sensitivity.py --save-traj` 추가해 둠 |
| A 재개 0% | **원인은 규명됨** — 재개 시점 손목 84도·위치 33cm 로 두 축이 다 범위 밖. 대책은 없음 |
| 전환 25% vs 39% (쥔 직후 vs 이동 중) | 방향은 CMI 와 맞지만 p=0.156 로 부족. 표본을 늘리면 확정 가능 |

## 7. 교훈 (반복하지 말 것)

1. **돌아가는 bash 스크립트를 덮어쓰지 말 것** — bash 는 파일을 조금씩 읽는다
2. `pgrep -f` 는 **자기 명령줄**과 래퍼 쉘까지 잡는다. `ps -eo pid,args` 로 실제 PID 확인
3. **한 축만 재고 결론짓지 말 것** — "회전 30도 절벽"이 조합 조건에 한 시간 만에 뒤집혔다
4. **예측을 먼저 적어 두면 검증이 빨라진다** — 판정 기준까지 스크립트에 넣으면 데이터가 오는 즉시 결론이 난다
5. ⭐ **자동 판정에는 유의성 조건을 넣을 것** — 주말 큐가 p=0.175 를 무시하고 점추정만 보고
   `cem_works` 로 진행해, 닫힌 길에 주말을 썼다
6. ⭐ **작은 표본의 +30%p 를 믿지 말 것** — 같은 쌍의 다른 에피소드 20회에서 −20%p 로 뒤집혔다
7. **가설을 검증할 때 집단이 맞는지 먼저 볼 것** — 정상 궤적으로 "틀어진 손목이 고쳐지는가"를
   잴 수 없다. 스크립트가 답을 뱉어도 그 답은 다른 질문의 답이다
8. **새 방법은 항등 검사부터** — w=1 에서 원본과 같아야 한다. 안 그러면 나중 차이가
   효과인지 버그인지 모른다. 실제로 첫 구현에 상쇄 오차가 있었다

## 8. 파일 지도

| 파일 | 내용 |
|---|---|
| `switch_experiment.py` | 전환 실험 본체 (`--cfg-w`, `--cfg-from` 추가됨) |
| `instr_cfg.py` | ⭐ 지시문 증폭 구현 |
| `test_cfg_identity.py` | w=1 항등 검사 |
| `analyze_cfg.py` | 증폭 평가 (성공률 + jerk) |
| `steerability.py` / `analyze_steer.py` | CMI 측정 |
| `rotation_at_resume.py` | 재개 시점 손목 회전 |
| `rotation_vs_travel.py` | 회전 드리프트 |
| `pose_sensitivity.py` | 자세 교란 (`--save-traj` 추가됨) |
| `analyze_lora_versions.py` | 원본/1차/2차/3차 비교 |
| `cem_noise.py`, `oracle_bon.py` | (닫힌 길) 노이즈 최적화·오라클 |

## 2026-09-28 14:50 — 목표: 모든 정확도 85% 이상 (사용자 지시)

- 원래 체크포인트 한계: 공식 설정(n_action_steps=1)으로도 LIBERO-Goal 81% (lerobot#2354). 혼자 할 때 T0 60, T3 50, T6 50, T9 50%
  → 전환 85% 전에 기본 실력부터 올려야 함. 방법: 약한 태스크마다 스크립트 전문가(≈100%) → 시범으로 학습(DAgger)
- 새 파일 `sim_only.py`: 모델 없이 시뮬레이터만 (SimRunner, episode(task, i))
- 조사 결과 (서랍): 미닫이 관절 `wooden_cabinet_1_{top,middle}_level`, 열면 +y, 범위 0.16m.
  손잡이 geom 가운데 `wooden_cabinet_1_g29`(z 1.015), 위 `g18`(z 1.089), 캐비닛 기준 [0.003, 0.102, 0.110/0.184], 에피소드마다 ±1cm
  정책 방식: 그리퍼 벌린 채 뒤 손가락을 손잡이 뒤에 걸어 +y로 당김
   - 위 서랍(T3): 손 방향 기본(euler [180,0,92]), 손잡이 기준 [0.003, +0.04, +0.02]에서 +y로 0.17m
   - 가운데 서랍(T0): 위 손잡이를 피하려 euler [-128,0,-177]로 기울여, 손잡이 기준 [0.005, -0.013, +0.02]에서 +y로 0.11m
- 다음: `task_experts.py`(새 파일, scripted_expert.py는 돌고 있는 파이프라인이 쓰므로 수정 금지)에
  open_drawer(ep, which) → T0/T3 시험(에피소드 0~9) → 와인병(T2,T9)·치즈(T6)·접시 밀기(T5) 전문가 → v6 데이터
- LoRA 5차(run_lora_v5.sh RESUME=1, run_v5_lane2.sh)는 계속 돌고 있음. 내일 analyze_v5.py

## 2026-09-28 17:15 — 서랍 전문가 완성 (task_experts.py)
- open_drawer_press(ep, "top"): 벌린 손, 손잡이 기준 [0.007, 0.047, 0.017], 뒤 손끝으로 막대 윗면 눌러 +y 당김 → 10/10
- open_middle_hook(ep): 손 방향은 demos T0 ep102 step104, 손잡이 기준 [-0.005,-0.011,0.015], 손끝을 막대 뒤 틈에 걸어 당김 → 20/20
- 앞의 open_drawer / open_drawer_pinch / open_drawer_hook 는 실패한 시도 (0/10) — 기록용
- 다음: 전환 상태에서 시험 → T3 전체(서랍+그릇) → 와인병·치즈·접시 → 6차 데이터

## 2026-09-28 17:35 — T3 전문가 (task_experts.drawer_bowl, test_t3.py)
- open_top_to(ep, 0.10): 위 서랍을 10cm만 (끝까지 열면 앞판이 그릇 위로 와서 손이 못 내려감)
- grasp_front: 그릇 앞쪽(+y) 테두리부터. 서랍 앞 높은 곳을 거쳐 수직 하강
- 결과 22~24/30. 실패 ep 9,12,13,15,24 (그릇이 캐비닛 가까이, 손이 앞판에 걸림), ep 4,19,21 (그릇이 서랍 길목, 서랍이 안 열림)
- 다음: 그릇을 먼저 옆으로 치우기 / 옆 테두리 잡기 → 90% 이상. 그 뒤 와인병·치즈·접시

## 2026-10-01 16:30 — 6차: 전부 전문가 방식
- 5차 실패 (75→39%). 원인: 스크립트 동작과 원래 모델 동작이 같은 지시문에 섞임 (eval_loss.py: 둘 다 배움)
- 10개 태스크 전문가 완성 task_experts.EXPERT (합계 296/300). test_all_experts.py <태스크> 30
- 6차 데이터: run_v6_collect.sh (정상 data/v6_normal, 전환·재개 data/v6_switch). 이어서 모으기 가능
- ⚠️ 메모리: 수집기 1개 2.4GB, PC 31GB·스왑 0 → 16개 동시에 재부팅됨. MAXP=5 + mem_guard.sh (5GB 아래면 수집기 멈춤)
- run_v6.sh: 수집 끝나면 학습(--full-expert --aug --workers 0, 24000스텝, outputs/v6_model) → 평가(장면 20~49, 30회: v6eval_forget, v6eval_switch)
- ⚠️ pkill -f 를 그 패턴이 들어간 명령줄에서 쓰지 말 것 (자기 셸까지 죽음). ps 로 PID 골라 kill

## 2026-10-01 17:50 — 밤샘·주말 자동 순서 (setsid 로 띄워 세션이 끊겨도 돈다)
1. run_v6_collect.sh (MAXP=5) — 시범 수집. mem_guard.sh 가 남은 메모리 5GB 아래면 수집기를 멈춤
2. run_v6.sh — 수집 끝나면 6차 학습(outputs/v6_model) → 6차 평가(outputs/v6eval_forget, v6eval_switch, 장면 20~49 각 30회)
3. run_v6_base.sh — 원래 모델 장면 30~49 평가 (v6eval_*_base)
4. run_v6_latency.sh — 지연 11스텝 조건 (v6lat_*_{v6,base}, 장면 20~29)
5. run_dagger1.sh — 6차 평가 뒤 DAgger 수집(data/v6_dagger1) → 3·4 끝나면 6.1차 학습(outputs/v61_model) → 평가(v61eval_*)
판정: .venv/bin/python analyze_v6.py  (성공률·95% 구간·85% 달성, 재개는 충돌 없는 6조합, 자연스러움)
오늘 고친 전문가: 7번 관절 한계(_aim, rotate_staged), 와인병 비스듬히 출발, 정밀 구간 DART 제외(se.PRECISE),
  서랍 일이면 그릇을 서랍 앞 비켜 내려놓기. A8→B0 재개는 충돌(열린 서랍이 접시를 덮음)이라 평가 제외.

## 2026-10-01 20:30 — 순서 변경 (run_v6_next.sh 하나로)
- 6차 4000스텝 미리 보기: 가운데 서랍 0/10, 그릇→접시 1/10, 접시 밀기 0/10 (큰 동작은 배웠으나 정밀 동작 미숙, 학습량 2.4%)
- run_v6.sh / run_v6_base.sh / run_v6_latency.sh / run_dagger1.sh 는 멈췄다 (학습 python 은 계속)
- run_v6_next.sh: 24000 끝 → 짧은 평가(v6q_*) → 6b차 36000스텝(outputs/v6b_model) → 정식 평가(v6eval_*) → DAgger(data/v6_dagger1) + 원래 모델(v6eval_*_base) → 6c차 12000스텝(outputs/v6c_model) → 평가(v6ceval_*) + 지연(v6lat_*)
