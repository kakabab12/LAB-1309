# 실습 1-2 — CNN, ResNet, ViT, 사전학습 (코랩 노트북)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kakabab12/LAB-1309/blob/main/04_%EC%8B%A4%EC%8A%B5/03_cnn_vit/LAB-1309_1-2_CNN_ViT.ipynb)

위 버튼을 누르면 코랩에서 바로 열립니다. **런타임 → 런타임 유형 변경 → T4 GPU** 로 바꾸고 위에서부터 실행하세요. 전체 약 25분.

## 📌 쉬운 말 요약

[01_딥러닝기초 1-2](../../01_공부순서/01_딥러닝기초.md) 의 네 가지 주제를 코드로 직접 확인합니다.

| 과제 | 주제 | 확인할 질문 |
|---|---|---|
| 1 | 합성곱 크기 계산 | 공식과 실제 결과가 같은가? 합성곱은 왜 파라미터가 적은가? |
| 2 | 작은 CNN | CIFAR-10 을 얼마나 맞히나? |
| 3 | **skip connection** | 같은 깊이(약 30층)에서 skip 이 없으면 학습이 얼마나 안 되나? |
| 4 | **ViT 직접 구현** | 이미지를 패치 토큰으로 자르면? CNN 보다 왜 낮게 나오나? |
| 5 | **사전학습** | 얼린 인코더 + Linear 한 층이 처음부터 학습한 모델을 이기나? |
| 6 | 비교 | 정확도·학습 파라미터·시간 표와 그래프 |

## 이전 실습과 이어지는 점

- [Transformer 실습](../02_transformer/2026-09-16_실습과제.md)에서 배운 **학습/검증 분리**와 **검증이 가장 좋을 때 저장(조기 종료)** 을 그대로 씀
- 과제 4 의 ViT 블록은 Transformer 실습과 같은 구조. 차이는 **causal mask 가 없다**는 것 (이미지 패치는 서로 다 봄)

## 연구와의 연결

| 이 실습 | SmolVLA / 우리 연구 |
|---|---|
| ViT 패치 임베딩 | SmolVLA 는 512×512 이미지를 패치로 잘라 SigLIP 비전 인코더에 넣음 |
| 사전학습 인코더 얼리기 (과제 5) | SmolVLA `freeze_vision_encoder=True`, 우리 LoRA 도 action expert 만 학습 |
| skip connection | Transformer 블록의 `x + attn(ln(x))` 도 같은 원리 |

## 파일

| 파일 | 내용 |
|---|---|
| `LAB-1309_1-2_CNN_ViT.ipynb` | 코랩 노트북 |
| `make_notebook.py` | 노트북 생성기. `QUICK=1` 로 가짜 데이터 빠른 점검 가능 |

> 노트북 코드는 로컬에서 가짜 데이터(FakeData)로 처음부터 끝까지 실행해 오류가 없음을 확인했습니다. CIFAR-10 실제 정확도는 직접 돌려서 기록하세요.
