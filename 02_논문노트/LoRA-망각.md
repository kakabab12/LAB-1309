# LoRA 와 망각 — LoRA (2021) / LoRA Learns Less and Forgets Less (2024)

## LoRA: Low-Rank Adaptation of Large Language Models

- **저자 / 연도 / 학회**: Edward J. Hu 외 (Microsoft) / 2021 / ICLR 2022
- **링크**: https://arxiv.org/abs/2106.09685
- **핵심**: 큰 가중치 W 는 얼리고 `W + B·A` (A, B 는 폭 r 의 작은 행렬)만 학습. 학습 후 합치면 추론 비용 증가 없음

## LoRA Learns Less and Forgets Less

- **저자 / 연도 / 학회**: Dan Biderman 외 / 2024 / TMLR 2024
- **링크**: https://arxiv.org/abs/2405.09673
- **읽은 날짜**: 2026-09-17 (초록 기준)
- **핵심 결과**
  - LoRA 는 전체 파인튜닝보다 **새 과제를 덜 배우지만, 원래 능력을 덜 잊는다**
  - 망각 방지 효과가 weight decay, dropout 보다 크다
  - 전체 파인튜닝이 만드는 변화의 rank 는 보통 LoRA 설정보다 10~100배 크다

## 내 연구와의 관계

- **가져온 것 (LoRA)**: SmolVLA action expert 에 LoRA 적용 (`train_lora.py`). 1080 Ti 에서 4.92M 파라미터(0.81%), VRAM 2.67GB
- **가져온 것 (망각 논문)**: LoRA 1차에서 태스크 1이 100% → 60% 로 **잊는 현상**이 나와서, 2차에서 **rank 16 → 8, 학습률 1e-4 → 5e-5** 로 낮춤. 여기에 정상 궤적을 섞는 **리허설**도 추가
- 주의: rank 를 낮추면 새로 배우는 양도 줄어든다 — 전환 성공률 향상과 망각 사이의 균형을 봐야 함
