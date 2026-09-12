# LAB 04. Action Chunking과 전환 — 연구의 심장

> 목표: 연구에서 제안할 **전환 방법을 직접 구현**해보기
> 소요: 4~6시간

## 먼저 이해할 것

```
일반 정책:   매 스텝마다 추론 -> 동작 1개
             (추론이 느리면 로봇이 멈춤)

Action Chunk: 한 번 추론 -> 동작 50개를 한꺼번에
             (추론하는 동안에도 로봇은 남은 동작을 실행)
```

| 설정 | 의미 | 값이 크면 |
|---|---|---|
| `chunk_size` | 한 번에 예측하는 스텝 수 | 부드럽지만 과거 정보로 미래를 예측 |
| `n_action_steps` | chunk 중 실제 실행하는 수 | **반응이 느려짐** (전환이 늦음) |

⭐ 예: chunk 50개를 뽑고 `n_action_steps=50`이면, 지시를 바꿔도 **50스텝 뒤에야** 반영됩니다.

---

## 실습 4-1. chunk 버퍼 구현

`ex15_chunk_buffer.py`

```python
import numpy as np

class ChunkBuffer:
    """VLA가 출력한 action chunk를 담아두고 하나씩 꺼내 쓴다."""

    def __init__(self, n_action_steps=10):
        self.n_action_steps = n_action_steps
        self.queue = []            # 실행 대기 중인 동작들
        self.steps_since_infer = 0

    def need_inference(self):
        return len(self.queue) == 0

    def push(self, chunk):
        """새 chunk (T, D)를 큐에 넣는다. n_action_steps만큼만 실행."""
        self.queue = list(chunk[:self.n_action_steps])

    def pop(self):
        return self.queue.pop(0) if self.queue else None


# 사용 예 (가짜 정책으로)
def fake_policy(obs, instruction):
    base = 0.1 if instruction == 'A' else -0.1
    return np.tile([base] * 7, (50, 1))      # (50, 7) chunk

buf = ChunkBuffer(n_action_steps=10)
instruction = 'A'

for t in range(30):
    if t == 12:
        instruction = 'B'                     # 여기서 지시 변경!
        print(f"--- t={t}: 지시를 B로 변경 ---")

    if buf.need_inference():
        buf.push(fake_policy(None, instruction))
        print(f"t={t}: 추론 (지시={instruction})")

    action = buf.pop()
    print(f"t={t:2d} action[0]={action[0]:+.2f}")
```

**관찰할 것**: t=12에 지시를 바꿨는데 **실제 동작은 언제 바뀌나요?**
→ 큐가 빌 때까지 기다립니다. 이것이 **전환 지연**의 정체입니다.

---

## 실습 4-2. 전환 전략 3가지 구현

`ex16_switch.py`

```python
import numpy as np

def strategy_wait(buf, new_chunk):
    """① 그냥 두기 - 큐가 빌 때까지 이전 동작 계속"""
    return                                  # 아무것도 안 함

def strategy_flush(buf, new_chunk):
    """② 즉시 교체 - 남은 걸 버리고 새 chunk로"""
    buf.queue = list(new_chunk[:buf.n_action_steps])

def strategy_blend(buf, new_chunk, k=8):
    """③ 블렌딩 - 겹치는 구간을 가중평균으로 섞기"""
    old = np.array(buf.queue) if buf.queue else None
    new = np.array(new_chunk[:buf.n_action_steps])
    if old is None or len(old) == 0:
        buf.queue = list(new)
        return
    n = min(k, len(old), len(new))
    w = np.linspace(1.0, 0.0, n).reshape(-1, 1)     # 이전 -> 새 것
    merged = new.copy()
    merged[:n] = w * old[:n] + (1 - w) * new[:n]
    buf.queue = list(merged)
```

**비교 실험**: 세 전략으로 각각 궤적을 만들고 LAB 03의 `jerk_score`로 비교하세요.

| 전략 | 반응 속도 | 부드러움 | 예상 |
|---|---|---|---|
| ① 그냥 두기 | 느림 | 좋음 | 전환이 늦음 |
| ② 즉시 교체 | 빠름 | 나쁨 | 동작이 튐 |
| ③ 블렌딩 | 빠름 | 좋음 | **가장 좋을 것** |

→ 이 표를 **숫자로 채우는 것**이 연구의 첫 결과가 됩니다.

---

## 실습 4-3. temporal ensembling (ACT 방식)

ACT 논문이 쓰는 기법입니다. 여러 chunk의 예측을 **지수 가중평균**으로 합칩니다.

`ex17_ensemble.py`

```python
import numpy as np
from collections import defaultdict

class TemporalEnsembler:
    """같은 시각에 대한 여러 chunk의 예측을 가중평균한다."""

    def __init__(self, m=0.01):
        self.m = m                       # 작을수록 최신 예측을 중시
        self.preds = defaultdict(list)   # 시각 -> [예측들]

    def add(self, t0, chunk):
        for i, a in enumerate(chunk):
            self.preds[t0 + i].append(a)

    def get(self, t):
        arr = np.array(self.preds[t])
        if len(arr) == 0:
            return None
        w = np.exp(-self.m * np.arange(len(arr)))[::-1]   # 최신에 큰 가중치
        w = w / w.sum()
        return (w.reshape(-1, 1) * arr).sum(0)


ens = TemporalEnsembler()
ens.add(0, np.random.randn(50, 7))
ens.add(10, np.random.randn(50, 7))       # 10스텝 뒤 새 추론
print(ens.get(15).shape)                  # 두 chunk의 예측이 섞임
```

**생각해볼 것**: 지시가 바뀌었을 때 **이전 지시로 만든 예측까지 섞이면** 문제가 될까요? → 여기서 연구 아이디어가 나옵니다.

---

## 실습 4-4. ⭐ 추론 지연 흉내내기

**저사양 환경을 고성능 PC에서 재현하는 방법**입니다.

`ex18_latency.py`

```python
import time
import numpy as np

class DelayedPolicy:
    """추론에 지연이 있는 정책을 흉내낸다 (저사양 GPU 재현)."""

    def __init__(self, policy_fn, delay_ms=0):
        self.policy_fn = policy_fn
        self.delay_ms = delay_ms
        self.pending = None          # (완료 시각, chunk)

    def request(self, obs, instruction, now):
        chunk = self.policy_fn(obs, instruction)
        self.pending = (now + self.delay_ms / 1000.0, chunk)

    def poll(self, now):
        """준비된 chunk가 있으면 돌려주고 비운다."""
        if self.pending and now >= self.pending[0]:
            chunk = self.pending[1]
            self.pending = None
            return chunk
        return None
```

**실험 설계**

| 지연 | 해당하는 환경 |
|---|---|
| 0ms | 이상적 |
| 50ms | 고성능 GPU |
| 200ms | 중급 GPU |
| 500ms | 1080 Ti fp32 추정 |
| 1000ms | CPU 추론 |

각 지연에서 **전략 3가지 × 전환 성공률/저크**를 측정하면 표가 완성됩니다.

---

## 실습 4-5. "잡은 상태" 판정

`ex19_grasp.py`

```python
import numpy as np

def is_holding(gripper_history, closed_thresh=0.3, hold_frames=5):
    """그리퍼가 닫힌 상태로 유지되면 물체를 잡고 있다고 본다.

    gripper_history: 최근 그리퍼 값들 (0=닫힘, 1=열림 가정)
    """
    if len(gripper_history) < hold_frames:
        return False
    recent = np.array(gripper_history[-hold_frames:])
    return bool((recent < closed_thresh).all())


# 완전히 닫히지 않고 멈추면 = 무언가 물고 있음
def is_holding_object(gripper_value, fully_closed=0.05, closed_thresh=0.3):
    return fully_closed < gripper_value < closed_thresh
```

⚠️ 실제 값의 방향(0이 열림인지 닫힘인지)과 범위는 **환경마다 다릅니다.** LAB 05에서 직접 찍어보고 맞추세요.

⭐ 이 함수가 **"잡은 상태에서의 전환"** 실험의 기준점이 됩니다.

---

## 실습 4-6. ACT 학습해보기 (선택, 시간 있으면)

LeRobot으로 실제 chunk 정책을 학습시켜 봅니다. SmolVLA보다 가벼워 1080 Ti에서도 가능합니다.

```bash
lerobot-train --policy.type=act --dataset.repo_id=lerobot/libero --steps=20000 --batch_size=8 --output_dir=./outputs/act_test
```

학습 후 chunk를 뽑아 **시간에 따른 동작 변화를 그래프로** 그려보세요.

---

## 체크리스트

- [ ] chunk 버퍼를 직접 구현했다
- [ ] `n_action_steps`가 전환 지연을 만드는 것을 코드로 확인했다
- [ ] 전환 전략 3가지를 구현하고 저크를 비교했다
- [ ] temporal ensembling의 원리를 안다
- [ ] 추론 지연을 인위로 주입하는 코드를 만들었다
- [ ] 그리퍼 값으로 "잡은 상태"를 판정하는 함수를 만들었다
