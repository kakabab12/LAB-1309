# LAB 01. PyTorch 기초 — 직접 만들어보기

> 목표: 남의 코드 복붙 없이 **학습 루프를 처음부터 끝까지** 짤 수 있게 되기
> 소요: 3~5시간

## 준비

```bash
conda create -n lab python=3.12 -y
conda activate lab
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install matplotlib numpy
```

확인 (1080 Ti 기준 `True (6, 1)`이 나와야 정상)

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability())"
```

---

## 실습 1-1. 텐서 감각 익히기

`ex01_tensor.py`

```python
import torch

# 만들기
a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.randn(2, 3)
c = torch.zeros(2, 2)

print(a.shape, a.dtype, a.device)

# 연산
print(a @ torch.ones(2, 2))     # 행렬곱
print(a * 2)                     # 원소별
print(a.mean(), a.sum(), a.max())

# 모양 바꾸기 (여기서 대부분의 버그가 납니다)
x = torch.randn(8, 3, 256, 256)          # (배치, 채널, 높이, 너비)
print(x.shape)
print(x.flatten(1).shape)                # (8, 196608)
print(x.permute(0, 2, 3, 1).shape)       # (8, 256, 256, 3) = TF 순서
print(x[:, :, ::2, ::2].shape)           # 절반 크기로 샘플링

# 차원 추가/제거
y = torch.randn(7)
print(y.unsqueeze(0).shape)              # (1, 7)  배치 차원 추가
print(y.unsqueeze(0).squeeze(0).shape)   # (7,)

# GPU로 보내기
if torch.cuda.is_available():
    x = x.cuda()
    print(x.device)
```

**직접 해볼 것**
- 동작 chunk를 흉내낸 텐서 `(배치 4, 시간 50, 동작 7)`을 만들고
  - 시간 축 평균을 구해보기
  - 앞 10스텝만 잘라내기
  - 두 chunk를 가중평균으로 섞어보기 (LAB 04의 예고편)

---

## 실습 1-2. autograd — 미분이 자동으로 되는 원리

`ex02_autograd.py`

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x          # y = x^2 + 2x
y.backward()                 # dy/dx = 2x + 2
print(x.grad)                # tensor(8.)

# 기울기는 누적됩니다 - 매 스텝 초기화해야 하는 이유
x.grad.zero_()

# 추론할 때는 기울기 계산을 끕니다 (메모리 절약)
with torch.no_grad():
    z = x * 2
print(z.requires_grad)       # False
```

---

## 실습 1-3. 선형회귀 — 가장 작은 학습 루프

`ex03_linear.py`

```python
import torch

# 정답: y = 2x + 3
true_w, true_b = 2.0, 3.0
x = torch.randn(200, 1)
y = true_w * x + true_b + 0.1 * torch.randn(200, 1)   # 노이즈 추가

w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
optimizer = torch.optim.SGD([w, b], lr=0.1)

for step in range(200):
    pred = x * w + b
    loss = ((pred - y) ** 2).mean()      # MSE

    optimizer.zero_grad()                # 1) 기울기 초기화
    loss.backward()                      # 2) 미분
    optimizer.step()                     # 3) 갱신

    if step % 50 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}  w {w.item():.3f}  b {b.item():.3f}")

print(f"\n학습 결과: w={w.item():.3f} (정답 2.0), b={b.item():.3f} (정답 3.0)")
```

⭐ **이 세 줄(zero_grad → backward → step)이 모든 딥러닝 학습의 핵심**입니다. 외우지 말고 이해하세요.

---

## 실습 1-4. MNIST 분류기 — 전체 구조 익히기

`ex04_mnist.py`

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1) 데이터
tf = transforms.Compose([
    transforms.ToTensor(),                       # 0~255 -> 0~1
    transforms.Normalize((0.1307,), (0.3081,)),  # 정규화 (통계값 중요!)
])
train_ds = datasets.MNIST('./data', train=True, download=True, transform=tf)
test_ds = datasets.MNIST('./data', train=False, download=True, transform=tf)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
test_dl = DataLoader(test_ds, batch_size=256)

# 2) 모델
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)

model = Net().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# 3) 학습
for epoch in range(3):
    model.train()
    for i, (x, y) in enumerate(train_dl):
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 200 == 0:
            print(f"epoch {epoch} step {i} loss {loss.item():.4f}")

    # 4) 평가
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
    print(f"epoch {epoch} 정확도 {correct / len(test_ds) * 100:.2f}%")

# 5) 저장
torch.save(model.state_dict(), 'mnist.pt')
```

**직접 바꿔보며 관찰할 것**

| 실험 | 관찰 |
|---|---|
| batch_size 64 → 512 | `nvidia-smi`로 VRAM 변화, 속도 변화 |
| lr 1e-3 → 1e-1 | 학습이 발산하는 모습 |
| `Normalize` 제거 | 정확도가 떨어지는지 |
| `model.eval()` 빼기 | (Dropout/BN이 있을 때) 결과가 흔들림 |
| device를 'cpu'로 | 시간이 몇 배 느려지는지 |

---

## 실습 1-5. 속도와 메모리 측정

`ex05_bench.py`

```python
import time, torch

def bench(device, n=50):
    x = torch.randn(64, 3, 224, 224, device=device)
    m = torch.nn.Conv2d(3, 64, 3, padding=1).to(device)
    m(x)                                   # 워밍업 (첫 호출은 느림)
    if device == 'cuda':
        torch.cuda.synchronize()           # GPU는 비동기라 동기화 필요
    t = time.perf_counter()
    for _ in range(n):
        m(x)
    if device == 'cuda':
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000

print(f"CPU  {bench('cpu'):.1f} ms")
if torch.cuda.is_available():
    print(f"CUDA {bench('cuda'):.1f} ms")
    print(f"최대 VRAM {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
```

⚠️ **GPU 시간 측정에는 `torch.cuda.synchronize()`가 필수**입니다. 없으면 실제보다 훨씬 빠르게 측정됩니다. (나중에 VLA 추론 시간 잴 때도 똑같이 적용됩니다)

---

## 체크리스트

- [ ] 텐서 shape을 자유롭게 바꿀 수 있다
- [ ] zero_grad → backward → step 순서와 이유를 안다
- [ ] MNIST 학습 코드를 안 보고 작성할 수 있다
- [ ] batch_size와 VRAM의 관계를 확인했다
- [ ] GPU 시간 측정 시 synchronize가 필요한 이유를 안다
- [ ] 정규화 통계가 왜 중요한지 설명할 수 있다
