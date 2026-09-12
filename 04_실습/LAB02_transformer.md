# LAB 02. Transformer 직접 만들기

> 목표: VLA 논문의 구조도를 **읽을 수 있게** 되기
> 소요: 4~6시간

## 왜 직접 만드나

VLA는 전부 Transformer 기반입니다. `nn.TransformerEncoder`를 부르는 것만으로는 논문의 "cross-attention으로 동작 토큰이 이미지를 참조한다" 같은 문장이 안 읽힙니다. **한 번 직접 짜면 평생 읽힙니다.**

---

## 실습 2-1. attention 한 줄씩 구현

`ex06_attention.py`

```python
import torch
import torch.nn.functional as F

# 입력: 토큰 5개, 각 토큰은 64차원
B, T, D = 2, 5, 64
x = torch.randn(B, T, D)

# Q, K, V 만들기
Wq, Wk, Wv = (torch.nn.Linear(D, D) for _ in range(3))
q, k, v = Wq(x), Wk(x), Wv(x)          # 각각 (B, T, D)

# 1) 점수 계산: 어떤 토큰이 어떤 토큰을 봐야 하나
scores = q @ k.transpose(-2, -1)        # (B, T, T)

# 2) 스케일링 (차원이 크면 값이 커져 softmax가 뾰족해짐)
scores = scores / (D ** 0.5)

# 3) 확률로 변환
attn = F.softmax(scores, dim=-1)        # (B, T, T), 각 행의 합 = 1

# 4) V를 가중합
out = attn @ v                          # (B, T, D)

print("attention 행렬 (첫 배치):")
print(attn[0].round(decimals=2))
print("행 합:", attn[0].sum(-1))        # 전부 1.0
```

**이해 확인**: `attn[0][2]`는 무엇을 의미할까요?
→ 3번째 토큰이 다른 토큰들을 **각각 얼마나 참고하는지**의 비율입니다.

---

## 실습 2-2. Multi-Head Attention

`ex07_mha.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, context=None, mask=None):
        # context가 있으면 cross-attention, 없으면 self-attention
        B, T, D = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
                   for t in qkv]        # (B, heads, T, head_dim)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)

x = torch.randn(2, 10, 128)
mha = MultiHeadAttention(128, n_heads=8)
print(mha(x).shape)      # (2, 10, 128)
```

**직접 해볼 것**
- `n_heads`를 1, 4, 8로 바꿔가며 파라미터 수가 같은지 확인
- causal mask(하삼각 행렬)를 만들어 미래를 못 보게 해보기

---

## 실습 2-3. Transformer 블록 완성

`ex08_block.py`

```python
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))    # residual
        x = x + self.mlp(self.norm2(x))     # residual
        return x
```

**꼭 이해할 것**
- `x = x + ...` 형태(**잔차 연결**)가 왜 필요한가 → 층이 깊어져도 학습이 되게
- LayerNorm이 attention **앞**에 오는 구조(pre-norm)가 요즘 표준

---

## 실습 2-4. ViT — 이미지를 토큰으로

`ex09_vit.py`

```python
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """이미지를 패치로 잘라 토큰으로 만든다."""
    def __init__(self, img_size=224, patch=16, dim=384):
        super().__init__()
        self.n_patches = (img_size // patch) ** 2      # 14*14 = 196
        # 커널과 stride가 같은 conv = 겹치지 않게 자르기
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)

    def forward(self, x):                 # (B, 3, 224, 224)
        x = self.proj(x)                  # (B, dim, 14, 14)
        return x.flatten(2).transpose(1, 2)   # (B, 196, dim)

pe = PatchEmbed()
print(pe(torch.randn(2, 3, 224, 224)).shape)   # (2, 196, 384)
```

⭐ **이게 VLA 이미지 입력부의 원리입니다.** 카메라 이미지 한 장이 토큰 수백 개가 되고, 언어 토큰과 함께 Transformer에 들어갑니다.

**계산해보기**: 카메라 2대 × 196토큰 + 언어 20토큰 = 412토큰. attention은 토큰 수의 제곱에 비례하니 **카메라를 늘리면 느려지는 이유**를 알 수 있습니다.

---

## 실습 2-5. 문자 단위 생성 모델 (통합)

Karpathy의 "Let's build GPT" 영상을 따라 하며 위 블록들을 쌓아 작은 언어 모델을 만들어보세요.

- 자기회귀 생성이 **왜 느린지** 몸으로 느낄 수 있습니다 (한 글자씩 생성)
- VLA에서 "동작 토큰을 자기회귀로 출력하면 느리다"는 말이 바로 이해됩니다

---

## 실습 2-6. VLA 흉내내기 (미니 프로젝트)

이미지와 언어를 받아 7차원 동작을 내는 **아주 작은 VLA**를 만들어보세요.

```python
import torch, torch.nn as nn

class MiniVLA(nn.Module):
    def __init__(self, dim=256, action_dim=7, chunk=10):
        super().__init__()
        self.patch = PatchEmbed(img_size=64, patch=8, dim=dim)   # 64px 이미지
        self.lang = nn.Embedding(100, dim)                        # 가짜 토크나이저
        self.blocks = nn.ModuleList([TransformerBlock(dim, 8) for _ in range(4)])
        self.action_query = nn.Parameter(torch.randn(1, chunk, dim))
        self.head = nn.Linear(dim, action_dim)

    def forward(self, img, lang_ids):
        img_tok = self.patch(img)                       # (B, 64, dim)
        lang_tok = self.lang(lang_ids)                  # (B, L, dim)
        act_tok = self.action_query.expand(img.size(0), -1, -1)
        x = torch.cat([img_tok, lang_tok, act_tok], dim=1)
        for blk in self.blocks:
            x = blk(x)
        act = x[:, -act_tok.size(1):]                   # 동작 토큰 자리만
        return self.head(act)                            # (B, chunk, 7)

m = MiniVLA()
img = torch.randn(2, 3, 64, 64)
ids = torch.randint(0, 100, (2, 8))
print(m(img, ids).shape)        # (2, 10, 7) = 동작 chunk!
```

이 구조가 **실제 VLA의 축소판**입니다. SmolVLA는 여기에
(1) 사전학습된 큰 VLM 백본, (2) flow matching 동작 생성, (3) 로봇 상태 입력을 더한 것입니다.

---

## 체크리스트

- [ ] Q, K, V의 역할을 그림으로 설명할 수 있다
- [ ] self-attention과 cross-attention의 차이를 안다
- [ ] 잔차 연결과 LayerNorm이 왜 필요한지 안다
- [ ] 이미지가 토큰이 되는 과정을 설명할 수 있다
- [ ] 자기회귀 생성이 왜 느린지 안다
- [ ] 이미지+언어 → 동작 chunk를 내는 모델을 직접 만들어봤다
