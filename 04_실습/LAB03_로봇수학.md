# LAB 03. 로봇 수학 — 좌표, 회전, 궤적

> 목표: 로봇 논문의 수식과 데이터 형식을 읽을 수 있게 되기
> 소요: 3~4시간

## 준비

```bash
pip install numpy scipy matplotlib
```

---

## 실습 3-1. 회전 표현 변환

`ex10_rotation.py`

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

# 오일러각 (도 단위): z축 30도, y축 0도, x축 0도 회전
r = R.from_euler('xyz', [0, 0, 30], degrees=True)

print("회전행렬:\n", r.as_matrix().round(3))
print("쿼터니언:", r.as_quat().round(3))          # (x, y, z, w)
print("축-각:", r.as_rotvec().round(3))           # LIBERO가 쓰는 형식
print("오일러각:", r.as_euler('xyz', degrees=True).round(1))

# 회전 합성 (순서가 중요합니다!)
r1 = R.from_euler('z', 30, degrees=True)
r2 = R.from_euler('y', 45, degrees=True)
print("r1 후 r2:", (r2 * r1).as_euler('xyz', degrees=True).round(1))
print("r2 후 r1:", (r1 * r2).as_euler('xyz', degrees=True).round(1))   # 다름!

# 점 회전시키기
p = np.array([1, 0, 0])
print("회전된 점:", r.apply(p).round(3))
```

**직접 해볼 것**
- 짐벌락 확인: pitch를 90도로 두고 roll과 yaw를 바꿔보기 → 같은 결과가 나오는 구간 발견
- 축-각 벡터의 **길이가 회전 각도**임을 확인 (`np.linalg.norm(rotvec)`)

---

## 실습 3-2. 2링크 팔 순기구학

`ex11_fk.py`

```python
import numpy as np
import matplotlib.pyplot as plt

L1, L2 = 1.0, 0.8      # 링크 길이

def fk(t1, t2):
    """관절 각도 -> 끝단 위치"""
    x1, y1 = L1*np.cos(t1), L1*np.sin(t1)
    x2, y2 = x1 + L2*np.cos(t1+t2), y1 + L2*np.sin(t1+t2)
    return (x1, y1), (x2, y2)

# 여러 자세 그려보기
fig, ax = plt.subplots()
for t1 in np.linspace(0, np.pi/2, 5):
    (x1, y1), (x2, y2) = fk(t1, np.pi/4)
    ax.plot([0, x1, x2], [0, y1, y2], 'o-')
ax.set_aspect('equal'); ax.grid(True)
plt.savefig('fk.png')
print("fk.png 저장")
```

**직접 해볼 것**
- 관절 각도를 무작위로 1000번 뽑아 끝단 위치를 점으로 찍어보기 → **작업 공간(도달 가능 영역)**이 눈에 보입니다
- 링크 길이를 바꾸면 작업 공간이 어떻게 변하는지

---

## 실습 3-3. 역기구학 — 답이 두 개

`ex12_ik.py`

```python
import numpy as np

L1, L2 = 1.0, 0.8

def ik(x, y):
    """끝단 위치 -> 관절 각도 (해가 2개: 팔꿈치 위/아래)"""
    d = np.hypot(x, y)
    if d > L1 + L2 or d < abs(L1 - L2):
        return None                       # 닿지 않음
    cos_t2 = (d**2 - L1**2 - L2**2) / (2*L1*L2)
    t2_up = np.arccos(np.clip(cos_t2, -1, 1))
    t2_dn = -t2_up
    sols = []
    for t2 in (t2_up, t2_dn):
        t1 = np.arctan2(y, x) - np.arctan2(L2*np.sin(t2), L1 + L2*np.cos(t2))
        sols.append((t1, t2))
    return sols

print("해:", np.degrees(ik(1.2, 0.8)).round(1))
print("닿지 않는 위치:", ik(5.0, 5.0))       # None
```

**여기서 배울 것**: 실제 6축 로봇도 똑같습니다. **해가 여러 개이거나 없을 수 있고**, 그래서 끝단 명령을 줄 때 작업 공간을 제한해야 합니다.

---

## 실습 3-4. ⭐ jerk 계산 — 동작이 얼마나 튀는가

이 연구의 **핵심 평가 지표**입니다.

`ex13_jerk.py`

```python
import numpy as np
import matplotlib.pyplot as plt

def jerk(traj, dt=1/30):
    """궤적 (T, D) -> 저크 (T-3, D). 3번 미분한다."""
    vel = np.diff(traj, axis=0) / dt       # 속도
    acc = np.diff(vel,  axis=0) / dt       # 가속도
    jrk = np.diff(acc,  axis=0) / dt       # 저크
    return jrk

def jerk_score(traj, dt=1/30):
    """스칼라 하나로 요약 - 값이 작을수록 부드러움"""
    return float(np.sqrt((jerk(traj, dt) ** 2).sum(axis=1)).mean())

T = 90
t = np.linspace(0, 3, T)

# 부드러운 궤적
smooth = np.stack([np.sin(t), np.cos(t), t/3], axis=1)

# 중간에 툭 끊기는 궤적 (전환 실패를 흉내)
jumpy = smooth.copy()
jumpy[45:] += np.array([0.3, -0.2, 0.1])     # 45프레임에서 순간이동

print(f"부드러운 궤적 저크: {jerk_score(smooth):.2f}")
print(f"튀는 궤적   저크: {jerk_score(jumpy):.2f}")

plt.plot(smooth[:, 0], label='smooth')
plt.plot(jumpy[:, 0], label='jumpy')
plt.axvline(45, color='r', ls='--', label='전환 시점')
plt.legend(); plt.savefig('jerk.png')
```

⭐ **이 함수를 그대로 연구에 씁니다.** 전환 방법 A와 B의 궤적을 기록해서 저크를 비교하면 "어느 쪽이 더 부드러운가"를 숫자로 말할 수 있습니다.

---

## 실습 3-5. 궤적 블렌딩 미리보기

`ex14_blend.py`

```python
import numpy as np

def blend(old, new, k=10):
    """이전 궤적에서 새 궤적으로 k스텝에 걸쳐 부드럽게 넘어간다."""
    out = new.copy()
    w = np.linspace(1, 0, k).reshape(-1, 1)      # 1 -> 0 가중치
    n = min(k, len(old), len(new))
    out[:n] = w[:n] * old[:n] + (1 - w[:n]) * new[:n]
    return out

old = np.tile([0.0, 0.0, 0.0], (30, 1))
new = np.tile([0.3, -0.2, 0.1], (30, 1))

hard = new                      # 그냥 갈아끼우기
soft = blend(old, new, k=10)    # 10스텝에 걸쳐 섞기

from utils import jerk_score              # 아래 설명 참고
print("급전환 저크:", jerk_score(np.vstack([old, hard])))
print("블렌딩 저크:", jerk_score(np.vstack([old, soft])))
```

> ⚠️ **여러 파일에서 함수를 재사용할 때**: 스크립트를 그대로 `import` 하면 그 파일의
> 출력·그래프 코드까지 같이 실행됩니다. `utils.py`를 따로 만들어 함수만 넣거나,
> 각 스크립트의 실행 부분을 `if __name__ == "__main__":` 아래로 옮기세요.

```python
# utils.py - 앞으로 계속 쓸 함수 모음
import numpy as np

def jerk(traj, dt=1/30):
    vel = np.diff(traj, axis=0) / dt
    acc = np.diff(vel,  axis=0) / dt
    return np.diff(acc, axis=0) / dt

def jerk_score(traj, dt=1/30):
    return float(np.sqrt((jerk(traj, dt) ** 2).sum(axis=1)).mean())
```

**실측 결과 (직접 돌려본 값)**

| 궤적 | 저크 |
|---|---|
| 부드러운 궤적 | 1.03 |
| 45프레임에서 툭 끊긴 궤적 | 465.47 |
| 급전환 (그냥 갈아끼움) | 708.95 |
| **10스텝 블렌딩** | **78.77** |

→ 블렌딩만으로 저크가 **약 9배** 줄어듭니다.

⭐ **이게 연구에서 제안할 방법의 원형입니다.** 여기서는 궤적으로 했지만, 실제로는 VLA가 출력한 **action chunk 두 개**를 이렇게 섞습니다. → LAB 04로 이어집니다.

---

## 실습 3-6. LIBERO 데이터 형식 이해

LIBERO의 관측과 동작 구조를 손으로 정리해보세요.

| 항목 | 차원 | 내용 |
|---|---|---|
| `observation.state` | 8 | 끝단 위치 3 + 축-각 3 + 그리퍼 2 |
| `observation.images.image` | 256×256×3 | 정면 카메라 |
| `observation.images.image2` | 256×256×3 | 손목 카메라 |
| `action` | 7 | 끝단 변화량 6 + 그리퍼 1, 범위 -1~1 |

**질문에 답해보기**
1. action의 앞 3개와 뒤 3개는 각각 무엇인가?
2. "변화량(delta)"이라는 건 무슨 뜻인가? 절대 좌표와 뭐가 다른가?
3. 그리퍼 값이 어떻게 변하면 "물체를 잡았다"고 볼 수 있을까?

---

## 체크리스트

- [ ] 오일러각/쿼터니언/축-각을 서로 변환할 수 있다
- [ ] 회전 합성의 순서가 중요한 이유를 안다
- [ ] FK와 IK를 간단한 팔로 직접 구현해봤다
- [ ] IK의 해가 여러 개이거나 없을 수 있음을 확인했다
- [ ] **jerk 계산 함수를 만들었다** (연구에 그대로 사용)
- [ ] 궤적 블렌딩으로 저크가 줄어드는 것을 숫자로 확인했다
