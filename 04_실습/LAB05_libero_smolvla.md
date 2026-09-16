# LAB 05. LIBERO + SmolVLA 실전

## 📌 쉬운 말 요약

**뭐 하는 실습?** 진짜 VLA 모델(SmolVLA)을 가상 로봇에서 돌려봅니다.

**솔직히 말하면**: 이 단계는 **설치가 제일 힘듭니다.** 실제로 돌리는 건 명령어 한 줄이에요. 설치하다 하루 이틀 날려도 정상입니다.

**순서**
1. 우분투에서 프로그램 설치 (여기가 고비)
2. 명령어 한 줄로 가상 로봇 실행 → 로봇이 블록 집는 걸 봄
3. **점수 기록** (예: 10번 중 8번 성공 = 80%)
4. 코드를 뜯어보며 **"모델이 언제 불려가고, 동작이 어떻게 실행되는지"** 파악
5. ⭐ 중간에 **시키는 말을 바꿔보기** ← 연구의 시작

**중요한 안심 포인트**: 논문에 적힌 점수(92%)가 안 나와도 괜찮습니다. 다른 사람들도 81~83%밖에 안 나온다고 보고했어요. **우리는 같은 조건에서 방법끼리 비교하는 거라 상관없습니다.**

---

> 목표: 공개 체크포인트로 **실제 VLA를 돌리고 수치를 뽑기**
> 소요: 1~2주 (설치에서 대부분의 시간이 감)

> ⚠️ **설치 방법이 바뀌었습니다.** 아래 conda + Python 3.12 + LeRobot main 조합은 현재 동작하지 않습니다
> (main 브랜치가 Python ≥ 3.12를 요구하고, LIBERO 의존성과 충돌).
> **실제로 돌아간 명령은 [05_libero/실행방법.md](05_libero/실행방법.md) 를 보세요** (uv + Python 3.10 + LeRobot 0.4.4, 2026-09-16 검증).
> 5~7단계 실험은 [05_libero/switch_experiment/](05_libero/switch_experiment/README.md) 에서 이어집니다.

## 1단계. 설치

### 1-1. 환경 만들기

```bash
conda create -n vla python=3.12 -y
conda activate vla
```

### 1-2. PyTorch 먼저 (GPU에 맞춰서)

1080 Ti (Pascal)

```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
```

4070 Ti (Ada) — 최신 버전 사용 가능

```bash
pip install torch torchvision
```

확인

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
```

### 1-3. LeRobot + LIBERO

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[libero,smolvla]"
```

### 1-4. 렌더링 설정

```bash
export MUJOCO_GL=egl
echo 'export MUJOCO_GL=egl' >> ~/.bashrc
```

### 1-5. ⭐ PyTorch 버전 재확인

LeRobot 설치 중에 PyTorch가 최신으로 바뀌었을 수 있습니다. **1-2의 확인 명령을 다시 실행**하세요. 바뀌었다면 1-2를 다시 실행합니다.

---

## 2단계. 첫 실행 (작동 확인)

태스크 1개만 3번

```bash
lerobot-eval --policy.path=HuggingFaceVLA/smolvla_libero --env.type=libero --env.task=libero_goal --env.task_ids=[0] --eval.batch_size=1 --eval.n_episodes=3
```

동시에 다른 터미널에서 GPU 확인

```bash
watch -n 1 nvidia-smi
```

**기록할 것**

| 항목 | 값 |
|---|---|
| 오류 없이 완료? | |
| 성공률 | |
| VRAM 사용량 | |
| 3 에피소드 소요 시간 | |

---

## 3단계. 기준 점수 측정

LIBERO-Goal 전체 (태스크 10개 × 10에피소드 = 100회)

```bash
lerobot-eval --policy.path=HuggingFaceVLA/smolvla_libero --env.type=libero --env.task=libero_goal --eval.batch_size=1 --eval.n_episodes=10
```

> ⚠️ 논문 점수(LIBERO-Goal 92%)가 재현되지 않는다는 보고가 여럿 있습니다(81~83%).
> **원인이 밝혀지지 않았지만 연구에는 문제없습니다.** 우리는 같은 환경에서 방법끼리 비교하니까요.
> 다만 **내 환경의 기준 점수를 반드시 기록**해두세요.

기록 양식

```
날짜:
GPU:                    PyTorch:            LeRobot:
명령어:
LIBERO-Goal 성공률:      %
소요 시간:
```

---

## 4단계. ⭐ 데이터 형식 직접 확인

문서를 믿지 말고 **직접 찍어보세요.**

`inspect_env.py` (LeRobot 저장소 안에서)

```python
# LeRobot 버전마다 import 경로가 다를 수 있습니다.
# 오류가 나면 lerobot 폴더에서 grep -rn "def make_env" 로 찾으세요.
import numpy as np

# 1) 환경 하나 만들기 (실제 API는 설치한 버전 문서 확인)
# 2) reset 해서 관측 받기
# 3) 아래 항목들을 출력

def describe(obs):
    for k, v in obs.items():
        arr = np.asarray(v)
        print(f"{k:35s} shape={str(arr.shape):20s} dtype={arr.dtype} "
              f"min={arr.min():.3f} max={arr.max():.3f}")

# 확인 목표
#  observation.state           (8,)      끝단 위치3 + 축-각3 + 그리퍼2
#  observation.images.image    (256,256,3) uint8
#  observation.images.image2   (256,256,3) uint8
#  action                      (7,)      끝단 변화량6 + 그리퍼1, -1~1
```

**해볼 것**
1. 관측 이미지 2장을 PNG로 저장해서 눈으로 확인
2. 한 에피소드의 **동작 7차원을 전부 기록**해서 그래프로 그리기
3. ⭐ **그리퍼 값만 따로 그래프로** → 물체를 잡는 순간이 어디인지 확인
4. 상태 8차원 중 어느 것이 그리퍼인지 찾기

이 4번이 **"잡은 상태 판정"의 기준**이 됩니다.

---

## 5단계. ⭐⭐ 평가 루프 분석 (가장 중요)

`lerobot-eval`이 실제로 무엇을 하는지 코드를 따라가며 **그림으로 그리세요.**

```bash
cd lerobot
grep -rn "def eval_policy\|def rollout" src/ | head
```

**답을 찾아야 할 질문**

| # | 질문 | 찾는 곳 |
|---|---|---|
| 1 | 언어 지시(task 문자열)는 어디서 모델에 들어가나? | 관측 딕셔너리에 task 키 |
| 2 | 모델은 몇 스텝마다 호출되나? | `n_action_steps` 사용처 |
| 3 | chunk는 어디에 저장되나? | 정책 내부의 queue |
| 4 | 새 chunk가 나오면 이전 것은 어떻게 되나? | queue 갱신 코드 |
| 5 | 성공 판정은 누가 하나? | 환경의 done/success |
| 6 | 이미지 전처리는? | 리사이즈, 정규화 |

**결과물**: 흐름도 1장 (손그림도 좋습니다) → `04_실습/05_libero/` 에 저장

---

## 6단계. 추론 속도 측정

`bench_policy.py`

```python
import time, torch, numpy as np

def bench_policy(policy, obs, n=30):
    """정책 1회 추론 시간 측정"""
    policy.select_action(obs)                     # 워밍업
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        policy.select_action(obs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()              # ⭐ 필수
        ts.append((time.perf_counter() - t) * 1000)
    ts = np.sort(ts)
    return {
        'median_ms': float(ts[len(ts)//2]),
        'p90_ms': float(ts[int(len(ts)*0.9)]),
        'max_ms': float(ts[-1]),
        'vram_GB': torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else 0,
    }
```

**측정 표**

| 환경 | 중앙값 | p90 | VRAM |
|---|---|---|---|
| 1080 Ti (fp32) | | | |
| 4070 Ti (bf16) | | | |
| CPU | | | |

→ 이 수치가 **"저사양에서 되는가"** 연구의 근거가 됩니다.

---

## 7단계. 🎯 첫 전환 실험

**평가 루프를 복사해서 내 스크립트로 만들고**, 중간에 지시를 바꿉니다.

`switch_experiment.py` (설계 뼈대)

```python
SWITCH_RATIOS = [0.3, 0.5, 0.8]      # 에피소드 진행률 기준 전환 시점

def run_episode(env, policy, task_a, task_b, switch_step, strategy='flush'):
    obs = env.reset()
    instruction = task_a
    traj, switched = [], False

    for t in range(MAX_STEPS):
        if t == switch_step and not switched:
            instruction = task_b                  # ⭐ 지시 교체
            switched = True
            if strategy == 'flush':
                policy.reset_queue()              # 남은 chunk 버리기
            elif strategy == 'blend':
                policy.enable_blend(k=8)          # 블렌딩 켜기
            # strategy == 'wait' 이면 아무것도 안 함

        action = policy.select_action(obs, instruction)
        obs, reward, done, info = env.step(action)
        traj.append(action)
        if done:
            break

    return {
        'success_b': info.get('success', False),
        'traj': np.array(traj),
        'switch_step': switch_step,
    }
```

**측정할 것**

| 지표 | 계산 |
|---|---|
| B 성공률 | 성공 / 전체 |
| 반응 시간 | 전환 시점 이후 동작 방향이 바뀌기까지의 스텝 |
| 저크 | LAB 03의 `jerk_score(traj)` |
| 낙하 여부 | 그리퍼 상태 변화 + 물체 위치 |

**실험 조합**: 전략 3가지 × 전환 시점 3가지 × 태스크 쌍 5개 × 10회 = 450 에피소드

---

## 자주 나는 오류

| 증상 | 해결 |
|---|---|
| `MUJOCO_GL` 오류 | `export MUJOCO_GL=egl` |
| `no kernel image is available` | PyTorch를 cu126 + 2.7로 재설치 |
| 관측 key 오류 | `observation.images.*` 이름 규칙 확인 |
| 성능이 이상하게 낮음 | `--env.control_mode` 절대/상대 확인 |
| 재현 안 됨 | seed 고정, `--env.init_states=true`, `--env.hard_reset=true` |
| 렌더링 느림 | `--env.hard_reset=false` (단, 재현성은 약간 떨어짐) |

---

## 체크리스트

- [ ] LIBERO가 설치되고 태스크가 실행된다
- [ ] LIBERO-Goal 기준 점수를 기록했다
- [ ] 관측/동작 차원을 직접 출력해 확인했다
- [ ] 그리퍼 값 그래프로 잡는 순간을 찾았다
- [ ] 평가 루프 흐름도를 그렸다
- [ ] 추론 속도를 GPU별로 측정했다
- [ ] **지시를 중간에 바꾸는 실험을 돌리고 영상을 남겼다**
