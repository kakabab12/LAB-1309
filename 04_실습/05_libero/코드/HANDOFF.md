# 인계 메모 — 10/7 '잠시 멈춤' 절이 최신

## 10/7 17시 — 잠시 멈춤 (가장 최신, 이것부터)
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
