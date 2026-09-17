#!/usr/bin/env python
"""1-2 CNN과 ViT 실습 노트북 생성기. 셀 코드는 CELLS 에 있고, QUICK=1 로 로컬에서 빠르게 검증할 수 있다."""
import json
import sys
from pathlib import Path

MD, CODE = "markdown", "code"

CELLS = [
(MD, """# LAB-1309 실습 1-2 — CNN, ResNet, ViT, 사전학습

[01_딥러닝기초 1-2](https://github.com/kakabab12/LAB-1309/blob/main/01_공부순서/01_딥러닝기초.md) 실습입니다.

| 과제 | 배우는 것 | 예상 시간 (T4) |
|---|---|---|
| 1 | 합성곱 출력 크기 — kernel, stride, padding, pooling | 1분 |
| 2 | 작은 CNN 으로 CIFAR-10 분류 | 5분 |
| 3 | **skip connection 이 왜 필요한가** — 깊은 CNN vs ResNet | 6분 |
| 4 | **ViT 직접 구현** — 이미지를 패치 토큰으로 | 6분 |
| 5 | **사전학습의 힘** — 미리 학습된 인코더 + 선형 분류기 | 4분 |
| 6 | 결과 비교표 | - |

**런타임 → 런타임 유형 변경 → T4 GPU** 로 설정하세요.

지난 실습(Transformer)에서 배운 **학습/검증 분리**와 **조기 종료(검증이 가장 좋을 때 저장)** 를 이번에도 씁니다."""),

(CODE, """# 0. 준비 — 데이터와 공용 학습 함수
import os, time, copy, math
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

QUICK = os.environ.get('QUICK') == '1'   # 로컬 코드 점검용 (Colab 에서는 False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device, '| QUICK:', QUICK)
if device == 'cpu' and not QUICK:
    print('⚠️ GPU 런타임이 아닙니다. 런타임 → 런타임 유형 변경 → T4 GPU')

mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

if QUICK:
    full = torchvision.datasets.FakeData(512, (3, 32, 32), 10, transform=train_tf)
    test_set = torchvision.datasets.FakeData(128, (3, 32, 32), 10, transform=test_tf)
    classes = [str(i) for i in range(10)]
else:
    full = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=test_tf)
    classes = full.classes

n_val = len(full) // 10
train_set, val_set = random_split(full, [len(full) - n_val, n_val], generator=torch.Generator().manual_seed(0))
BS = 64 if QUICK else 256
train_loader = DataLoader(train_set, BS, shuffle=True, num_workers=0 if QUICK else 2, pin_memory=True)
val_loader = DataLoader(val_set, BS, num_workers=0 if QUICK else 2)
test_loader = DataLoader(test_set, BS, num_workers=0 if QUICK else 2)
print('학습', len(train_set), '| 검증', len(val_set), '| 테스트', len(test_set), '| 클래스', classes)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += F.cross_entropy(out, y, reduction='sum').item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, 100 * correct / total


def train(model, epochs, lr=1e-3, name='model', wd=5e-4):
    \"\"\"학습 + 검증 정확도가 가장 좋을 때의 가중치를 저장(조기 종료 아이디어). 기록을 돌려준다.\"\"\"
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    best_acc, best_state, hist = -1, None, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        run, n = 0.0, 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            run += loss.item() * y.size(0); n += y.size(0)
            if QUICK and i >= 2:
                break
        vl, va = evaluate(model, val_loader)
        hist.append({'epoch': ep, 'train_loss': run / n, 'val_loss': vl, 'val_acc': va})
        if va > best_acc:
            best_acc, best_state = va, copy.deepcopy(model.state_dict())
        print(f'[{name}] epoch {ep} | train loss {run / n:.3f} | val loss {vl:.3f} | val acc {va:.1f}% | {time.time() - t0:.0f}초')
    model.load_state_dict(best_state)
    _, test_acc = evaluate(model, test_loader)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f'[{name}] 최고 검증 {best_acc:.1f}% → 테스트 {test_acc:.1f}% | 학습 파라미터 {params:.2f}M | {time.time() - t0:.0f}초')
    return {'name': name, 'test_acc': test_acc, 'best_val': best_acc, 'params_M': params, 'sec': time.time() - t0, 'hist': hist}


results = []
EPOCHS = 1 if QUICK else 8"""),

(MD, """## 과제 1. 합성곱 출력 크기

합성곱을 지나면 이미지 크기가 이렇게 바뀝니다.

```
출력 크기 = floor( (입력 크기 + 2 × padding − kernel) / stride ) + 1
```

- **kernel**: 한 번에 보는 창 크기
- **stride**: 창을 몇 칸씩 옮기나 (2면 크기가 절반)
- **padding**: 가장자리에 0을 덧대서 크기를 유지
- **pooling**: 창 안에서 최댓값/평균만 남겨 크기를 줄임

먼저 손으로 계산해 보고, 코드로 확인하세요."""),

(CODE, """# 과제 1. 공식으로 계산한 값과 실제 PyTorch 결과가 같은지 확인

def out_size(n, k, s=1, p=0):
    return (n + 2 * p - k) // s + 1

x = torch.randn(1, 3, 32, 32)
cases = [
    ('3x3, stride 1, padding 1 (크기 유지)', nn.Conv2d(3, 16, 3, 1, 1), (32, 3, 1, 1)),
    ('3x3, stride 2, padding 1 (절반)', nn.Conv2d(3, 16, 3, 2, 1), (32, 3, 2, 1)),
    ('5x5, stride 1, padding 0 (줄어듦)', nn.Conv2d(3, 16, 5, 1, 0), (32, 5, 1, 0)),
    ('4x4 패치, stride 4 (ViT 패치 임베딩)', nn.Conv2d(3, 192, 4, 4, 0), (32, 4, 4, 0)),
    ('2x2 max pooling', nn.MaxPool2d(2), (32, 2, 2, 0)),
]
for name, layer, (n, k, s, p) in cases:
    real = layer(x).shape[-1]
    calc = out_size(n, k, s, p)
    print(f'{name:38s} 공식 {calc:2d} | 실제 {real:2d} | {\"✅\" if calc == real else \"❌\"}')

# 파라미터 수: 합성곱은 이미지 크기와 무관하다
conv = nn.Conv2d(3, 16, 3)
print('\\nConv2d(3→16, 3x3) 파라미터:', sum(p.numel() for p in conv.parameters()), '= 16 × (3×3×3) + 16(bias)')
fc = nn.Linear(3 * 32 * 32, 16)
print('같은 출력 16개를 Linear 로 만들면:', sum(p.numel() for p in fc.parameters()), '← 이미지가 커질수록 폭증')"""),

(MD, """## 과제 2. 작은 CNN 으로 CIFAR-10 분류

32×32 컬러 이미지 10종류(비행기, 자동차, 새, 고양이 …)를 분류합니다.

```
[conv-bn-relu] ×2 → maxpool   (32 → 16)
[conv-bn-relu] ×2 → maxpool   (16 → 8)
[conv-bn-relu] ×2 → maxpool   (8 → 4)
평균 풀링 → Linear(10)
```

**BatchNorm(bn)** 은 각 층 출력의 분포를 고르게 맞춰서 학습을 안정시킵니다."""),

(CODE, """# 과제 2. 작은 CNN

def conv_bn(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

class SmallCNN(nn.Module):
    def __init__(self, w=64):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn(3, w), conv_bn(w, w), nn.MaxPool2d(2),
            conv_bn(w, 2 * w), conv_bn(2 * w, 2 * w), nn.MaxPool2d(2),
            conv_bn(2 * w, 4 * w), conv_bn(4 * w, 4 * w), nn.MaxPool2d(2),
        )
        self.head = nn.Linear(4 * w, 10)

    def forward(self, x):
        x = self.features(x)                  # (B, 256, 4, 4)
        return self.head(x.mean(dim=(2, 3)))  # 전역 평균 풀링

results.append(train(SmallCNN(), EPOCHS, lr=3e-3, name='작은 CNN'))"""),

(MD, """## 과제 3. skip connection 이 왜 필요한가

층을 깊게 쌓으면 더 잘 될 것 같지만, **그냥 쌓기만 하면 오히려 학습이 안 됩니다.** 기울기가 수십 층을 거꾸로 지나오면서 사라지거나 뒤틀리기 때문입니다.

**ResNet** 의 해결책은 한 줄입니다.

```
plain:     x → [conv → conv] → 출력
residual:  x → [conv → conv] → + x → 출력     ← 입력을 그대로 더해줌 (skip connection)
```

입력을 그대로 더하면, 블록이 "아무것도 안 하기"(= 0 을 출력)만 배워도 원래 신호가 통과합니다. 기울기도 이 지름길로 곧장 흐릅니다. 그래서 층을 늘려도 **최소한 얕은 망만큼은** 학습됩니다.

**같은 깊이(conv 약 30층)** 로 두 가지를 만들어 비교합니다.

> 참고: BatchNorm 이 있으면 기울기가 완전히 사라지지는 않아서, 차이는 **학습 loss 가 얼마나 빨리·낮게 내려가는지**로 봐야 잘 보입니다. 첫 층 기울기 크기는 참고용으로만 출력합니다."""),

(CODE, """# 과제 3. 깊은 plain CNN vs ResNet (같은 깊이)

class Block(nn.Module):
    def __init__(self, c, residual):
        super().__init__()
        self.residual = residual
        self.body = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                                  nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        out = self.body(x)
        return F.relu(out + x) if self.residual else F.relu(out)

class DeepNet(nn.Module):
    def __init__(self, residual, blocks_per_stage=5, w=32):
        super().__init__()
        self.stem = conv_bn(3, w)
        layers, c = [], w
        for stage in range(3):
            if stage > 0:
                layers += [nn.Conv2d(c, 2 * c, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(2 * c), nn.ReLU(inplace=True)]
                c *= 2
            layers += [Block(c, residual) for _ in range(blocks_per_stage)]
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(c, 10)

    def forward(self, x):
        return self.head(self.layers(self.stem(x)).mean(dim=(2, 3)))

def first_layer_grad(model):
    \"\"\"배치 하나로 역전파했을 때 첫 합성곱 층 기울기 크기.\"\"\"
    model = model.to(device).train()
    x, y = next(iter(train_loader))
    loss = F.cross_entropy(model(x.to(device)), y.to(device))
    model.zero_grad(); loss.backward()
    return model.stem[0].weight.grad.norm().item()

for residual in [False, True]:
    name = 'ResNet (skip 있음)' if residual else 'plain (skip 없음)'
    torch.manual_seed(0)
    m = DeepNet(residual)
    g = first_layer_grad(DeepNet(residual))
    print(f'{name}: (참고) 초기 첫 층 기울기 크기 {g:.4f}')
    r = train(m, EPOCHS, lr=3e-3, name=name)
    r['first_grad'] = g
    results.append(r)"""),

(MD, """## 과제 4. ViT 직접 구현 — 이미지를 패치 토큰으로

ViT 는 CNN 을 쓰지 않습니다. 이미지를 **바둑판처럼 잘라서** 각 조각을 단어(토큰)처럼 Transformer 에 넣습니다.

```
32×32 이미지 → 4×4 패치 64개 → 각 패치를 192차원 벡터로 → [CLS] 토큰 추가 → 위치 임베딩 더하기
            → Transformer 블록 6개 → [CLS] 토큰으로 분류
```

- **패치 임베딩** = kernel 4, stride 4 합성곱 (과제 1에서 확인한 그것)
- **[CLS] 토큰**: 분류용 요약 토큰. 모든 패치를 어텐션으로 참고해 전체 정보를 모음
- 블록 구조는 지난 Transformer 실습과 같음. 차이는 **causal mask 가 없다** (이미지는 앞뒤 순서가 없으니 모든 패치를 서로 봄)

**예상**: 같은 시간 학습하면 **CNN 보다 낮게** 나옵니다. ViT 는 "가까운 픽셀끼리 관련 있다"는 가정이 없어서 데이터가 많이 필요합니다. 그래서 VLA 는 **대규모로 사전학습된 ViT(SigLIP)** 를 가져다 씁니다 → 과제 5"""),

(CODE, """# 과제 4. ViT

class PatchEmbed(nn.Module):
    def __init__(self, patch=4, dim=192):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, patch, patch)

    def forward(self, x):
        x = self.proj(x)                      # (B, dim, 8, 8)
        return x.flatten(2).transpose(1, 2)   # (B, 64, dim)

class ViTBlock(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.out = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.heads, self.dropout = heads, dropout
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim), nn.Dropout(dropout))

    def attn(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k, v = (t.view(B, N, self.heads, C // self.heads).transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)  # is_causal 없음
        return self.out(o.transpose(1, 2).reshape(B, N, C))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class ViT(nn.Module):
    def __init__(self, dim=192, depth=6, heads=6, patch=4):
        super().__init__()
        self.patch = PatchEmbed(patch, dim)
        n = (32 // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, n + 1, dim) * 0.02)
        self.blocks = nn.Sequential(*[ViTBlock(dim, heads) for _ in range(depth)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)

    def forward(self, x):
        x = self.patch(x)
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], 1) + self.pos
        x = self.ln(self.blocks(x))
        return self.head(x[:, 0])             # [CLS] 토큰

vit = ViT()
print('패치 토큰 shape:', vit.patch(torch.randn(2, 3, 32, 32)).shape)
results.append(train(vit, EPOCHS, lr=1e-3, name='ViT (처음부터)', wd=0.05))"""),

(MD, """## 과제 5. 사전학습의 힘 — 인코더는 얼리고 분류기만 학습

ImageNet(120만 장)으로 **미리 학습된 ResNet-18** 을 가져와서

- 인코더는 **완전히 얼리고** (학습 안 함)
- 마지막에 **Linear 한 층만** 새로 학습합니다 (linear probe)

학습하는 파라미터가 5천 개 남짓인데도 처음부터 학습한 모델과 비교해 보세요.

이게 **SmolVLA 가 이미지를 다루는 방식**과 같습니다. SmolVLA 의 비전 인코더(SigLIP)도 **얼려둔 채** 쓰고(`freeze_vision_encoder=True`), 그 위의 작은 부분만 학습합니다. 우리 연구의 LoRA 도 같은 발상입니다."""),

(CODE, """# 과제 5. 사전학습 ResNet-18 + linear probe

weights = None if QUICK else torchvision.models.ResNet18_Weights.IMAGENET1K_V1
backbone = torchvision.models.resnet18(weights=weights)
backbone.fc = nn.Identity()                        # 분류층 떼기 → 512차원 특징
for p in backbone.parameters():
    p.requires_grad_(False)                        # 얼리기

class Probe(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(512, 10)

    def forward(self, x):
        x = F.interpolate(x, size=112, mode='bilinear', align_corners=False)  # 사전학습 해상도에 가깝게
        with torch.no_grad():
            f = self.backbone.eval()(x)
        return self.head(f)

results.append(train(Probe(backbone), max(1, EPOCHS // 2), lr=3e-3, name='사전학습 + 선형 분류기', wd=0.0))"""),

(CODE, """# 6. 결과 비교
import matplotlib.pyplot as plt

print(f\"{'모델':24s} {'테스트 정확도':>10s} {'학습 파라미터':>12s} {'시간':>7s}\")
for r in results:
    print(f\"{r['name']:24s} {r['test_acc']:9.1f}% {r['params_M']:10.2f}M {r['sec']:6.0f}초\")

# Colab 에는 한글 폰트가 없어서 그래프 글자는 영어로
labels_en = {'작은 CNN': 'Small CNN', 'plain (skip 없음)': 'Plain (no skip)', 'ResNet (skip 있음)': 'ResNet (skip)',
             'ViT (처음부터)': 'ViT (scratch)', '사전학습 + 선형 분류기': 'Pretrained + linear'}
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for r in results:
    lab = labels_en.get(r['name'], r['name'])
    axs[0].plot([h['epoch'] for h in r['hist']], [h['train_loss'] for h in r['hist']], marker='o', label=lab)
    axs[1].plot([h['epoch'] for h in r['hist']], [h['val_acc'] for h in r['hist']], marker='o', label=lab)
axs[0].set_xlabel('epoch'); axs[0].set_ylabel('train loss'); axs[0].set_title('Train loss (plain vs ResNet here)'); axs[0].grid(alpha=0.3)
axs[1].set_xlabel('epoch'); axs[1].set_ylabel('val accuracy (%)'); axs[1].set_title('Validation accuracy'); axs[1].grid(alpha=0.3)
axs[1].legend(fontsize=8)
axs[2].bar(range(len(results)), [r['test_acc'] for r in results])
axs[2].set_xticks(range(len(results)), [labels_en.get(r['name'], r['name']) for r in results], rotation=20, ha='right', fontsize=8)
axs[2].set_ylabel('test accuracy (%)'); axs[2].set_title('Test accuracy')
plt.tight_layout(); plt.show()

deep = {r['name']: r for r in results if 'first_grad' in r}
if len(deep) == 2:
    print('\\n(참고) 초기 첫 층 기울기 크기:', {k: round(v['first_grad'], 4) for k, v in deep.items()})"""),

(MD, """# 확인할 것

1. **과제 1**: 공식과 실제가 모두 ✅ 인가? 합성곱 파라미터 수가 Linear 보다 얼마나 적은가?
2. **과제 3**: plain(skip 없음)이 ResNet 보다 **학습 loss 자체가 덜 내려가는가?** 과적합이 아니라 **학습이 안 되는** 것 — 이게 skip connection 이 필요한 이유
3. **과제 4**: ViT(처음부터)가 CNN 보다 낮은가? 왜 그럴까?
4. **과제 5**: 사전학습 + Linear 한 층이 처음부터 학습한 ViT 를 이기는가? 학습 파라미터 수를 비교해 보기

# 더 해볼 것

- ViT 의 패치 크기 4 → 8 로: 토큰이 64 → 16 개로 줄면 속도·정확도가 어떻게 되나 (어텐션 계산량은 토큰 수의 제곱)
- 과제 5 에서 인코더를 **얼리지 않고** 같이 학습(파인튜닝): 정확도는 오르지만 시간이 얼마나 늘어나나
- DeepNet 의 `blocks_per_stage` 를 3 → 6 으로: plain 은 더 나빠지고 ResNet 은 버티는가

# VLA 와의 연결

| 이 실습 | SmolVLA / 우리 연구 |
|---|---|
| ViT 패치 임베딩 | SmolVLA 는 512×512 이미지를 패치로 잘라 SigLIP 비전 인코더에 넣음 |
| [CLS] / 패치 토큰 | 이미지 토큰들이 언어 토큰과 함께 VLM 에 들어감 |
| 사전학습 인코더 얼리기 | SmolVLA `freeze_vision_encoder=True`, 우리 LoRA 도 action expert 만 학습 |
| skip connection | Transformer 블록의 `x + attn(ln(x))` 도 같은 원리 |"""),
]


def build():
    nb = {"nbformat": 4, "nbformat_minor": 0,
          "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}, "accelerator": "GPU"},
          "cells": []}
    for kind, src in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == CODE:
            cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    return nb


if __name__ == "__main__":
    out = Path(__file__).parent / "LAB-1309_1-2_CNN_ViT.ipynb"
    out.write_text(json.dumps(build(), ensure_ascii=False, indent=1))
    if "--export-py" in sys.argv:
        py = "\n\n".join(src for kind, src in CELLS if kind == CODE)
        Path(sys.argv[sys.argv.index("--export-py") + 1]).write_text(py)
    print("저장:", out)
