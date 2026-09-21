# 실험 코드 사본

여기 있는 파일은 **연구실 PC `~/smolVLA/` 에 있는 실제 실험 코드의 사본**입니다.
다른 PC에서 이어받을 때 이걸 복사해 쓰시면 됩니다.

```bash
git clone https://github.com/kakabab12/LAB-1309.git
cp LAB-1309/04_실습/05_libero/코드/*.py ~/smolVLA/
cp LAB-1309/04_실습/05_libero/코드/*.sh ~/smolVLA/
chmod +x ~/smolVLA/*.sh
```

환경 설치와 사용법은 **[이어받기.md](../../../이어받기.md)** 를 보세요.

⚠️ 사본이라 연구실 PC 쪽이 더 최신일 수 있습니다. 실험을 돌리기 전에
`~/smolVLA/` 쪽과 날짜를 비교해 보세요.

---

## 무엇을 돌릴 때 무엇을 쓰나

### 전환 실험 (본체)

| 파일 | 내용 |
|---|---|
| **`switch_experiment.py`** | 모든 전환 전략이 들어 있는 본체. 대부분의 실험이 이걸 부릅니다 |

```bash
.venv/bin/python switch_experiment.py --task-a 8 --task-b 7 \
  --switch-at grasp:3 --strategy flush --episodes 10 --out outputs/switch
```

주요 옵션

| 옵션 | 뜻 |
|---|---|
| `--task-a` / `--task-b` | 원래 하던 일 / 새로 시킨 일 (태스크 번호는 [이어받기.md](../../../이어받기.md) 7-3) |
| `--switch-at grasp:3` | 물체를 쥐고 3스텝 뒤에 지시를 바꿈 |
| `--strategy flush` | 남은 동작을 버리고 즉시 새 지시로 (기준선) |
| `--cfg-w 1.5` | **지시문 증폭.** 1.0 이면 원본과 완전히 같음 |
| `--instr-repeat 3` | **지시문을 3번 반복해 넣음** (대안 증폭) |
| `--episodes 10` | 반복 횟수 |

⚠️ `--strategy retreat` / `ret_pos` / `ret_rot` 는 **초기 자세로 되돌리므로 제약 위반**입니다.
상한을 재는 용도로만 쓰세요.

### 조종 가능성(CMI) 측정

| 파일 | 내용 |
|---|---|
| `steerability.py` | CMI 측정 — **같은 상태에서 지시문만 바꿔** 동작이 얼마나 달라지는지 |
| `analyze_steer.py` | 결과 표·그래프 |
| `compare_steer.py` | 여러 조건(증폭 켬/끔)의 CMI 비교 |
| `cmi_curve.py` | CMI 시간 곡선 — 귀가 언제 닫히고 열리는가 |
| `pair_steer.py` | 쌍별 조종 가능성이 성공률을 예측하는가 |

```bash
.venv/bin/python steerability.py --tasks 8 4 1 2 --episodes 5 --k 8 --out outputs/steerability
.venv/bin/python analyze_steer.py
```

### 지시문 증폭 (새 방법)

| 파일 | 내용 |
|---|---|
| `instr_cfg.py` | 구현 — 이전 지시를 **음의 조건**으로 써서 새 지시를 키움 |
| `test_cfg_identity.py` | **항등 검사** — w=1 에서 원본과 비트 단위로 같은지 |
| `analyze_cfg.py` | 평가 — 성공률 **+ jerk(자연스러움)** 를 항상 같이 |

> 💡 `test_cfg_identity.py` 는 **새 방법을 만들 때의 본보기**입니다.
> 파라미터를 중립값으로 두면 원본과 같아야 합니다. 안 그러면 나중에 나오는 차이가
> 효과인지 버그인지 알 수 없습니다. 실제로 첫 구현에 상쇄 오차가 있었습니다.

### 자세·회전 분석

| 파일 | 내용 |
|---|---|
| `pose_sensitivity.py` | 팔을 옮기거나 손목을 돌려 놓고 시작했을 때의 성공률 |
| `analyze_boundary.py` | 교란 경계 곡선 |
| `rotation_at_resume.py` | A 재개 시점의 손목 회전 (84도가 여기서 나왔습니다) |
| `rotation_vs_travel.py` | 손목 회전이 시간에 따라 어떻게 변하는가 |

### 닫힌 길 (기록용, 다시 열지 마세요)

| 파일 | 왜 닫혔나 |
|---|---|
| `oracle_bon.py` | 후보를 여러 개 뽑아 고르는 상한. 판정기로는 못 고른다는 결론 |
| `cem_noise.py` | 노이즈 평균이동 최적화. 검증 50회에서 −10%p (p=0.885) |
| `analyze_lora_versions.py` | LoRA 1·2·3차 비교. 32/31/30% 로 원본과 같음 |

### 큐 스크립트

GPU 가 1개라 **순서대로 이어 달리게** 만든 것들입니다.
첫 인자로 PID 를 주면 그 프로세스가 끝날 때까지 기다립니다.

```bash
PID=$(ps -eo pid,args | awk '$2=="bash" && /run_repeat\.sh/{print $1; exit}')
nohup ./run_내실험.sh "$PID" > outputs/내로그.log 2>&1 &
```

| 파일 | 내용 |
|---|---|
| `run_cfg.sh` | 지시문 증폭 w 스윕 |
| `run_cfg_cmi.sh` | 증폭이 CMI 를 올리는지 |
| `run_cmi_curve.sh` | CMI 시간 곡선 |
| `run_manypairs.sh` | 쌍을 4개 → 20개로 |
| `run_repeat.sh` | 지시문 토큰 반복 |

⚠️ **돌아가는 중인 `.sh` 파일을 편집하지 마세요.** bash 는 스크립트를 조금씩 읽어서,
실행 중에 바꾸면 엉뚱한 곳부터 이어 읽습니다. 실제로 한 번 깨졌습니다.
