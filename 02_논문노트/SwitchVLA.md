# SwitchVLA: Execution-Aware Task Switching for Vision-Language-Action Models

- **저자 / 연도 / 학회**: Meng Li, Zhen Zhao, Zhengping Che, Fei Liao, Kun Wu, Zhiyuan Xu, Pei Ren, Zhao Jin, Ning Liu, Jian Tang / 2025-06 / arXiv preprint
- **링크**: https://arxiv.org/abs/2506.03574 · 프로젝트 https://switchvla.github.io/
- **코드**: 공개 여부 확인 안 됨 (프로젝트 페이지에 언급 없음)
- **읽은 날짜**: 2026-09-16 (초록·프로젝트 페이지 수준, 본문 정독 필요)

> ⚠️ **우리 연구 주제와 거의 같은 논문입니다.** 태스크 A 수행 중 B 지시가 들어오는 상황을 다루고, 시뮬레이션도 **LIBERO-Goal**을 씁니다.

## 1. 이 논문이 푸는 문제 (한 문장)

기존 VLA는 태스크 의도가 고정돼 있다고 가정해서, 수행 도중 새 지시가 들어오면 반응하지 못한다.

## 2. 핵심 아이디어 (한 문장)

전환을 "새 태스크를 처음부터 하는 문제"가 아니라 **실행 상태(execution state)에 따라 행동을 조절(behavior modulation)하는 문제**로 보고, 시연을 접촉 단계로 나눠 학습한다.

## 3. 방법

- 전문가 시연을 **접촉 단계(contact phase)** 로 분할 → 정책이 현재 태스크 진행 정도를 추론
- 진행 정도와 지시 맥락에 따라 여러 행동 모드를 내는 **multi-behavior conditional policy**
- 외부 플래너나 전환 전용 데이터 없이 동작한다고 주장

## 4. 실험

- 벤치마크: **LIBERO-Goal** 8개 태스크(태스크당 시연 50개) + 실물 Franka Panda 양팔 2대
- 전환 시점을 early(접촉 전) / mid(접촉 중) / late(동작 후) 세 가지로 나눠 쌍 실험
- 주요 수치 (LIBERO-Goal 성공률, No switch / Early / Mid / Late)

| 방법 | No switch | Early | Mid | Late |
|---|---|---|---|---|
| π0 | 92.3 | 40.7 | 8.3 | 10.2 |
| OpenVLA-OFT | 98.0 | 40.6 | 11.1 | 13.0 |
| **SwitchVLA** | 93.0 | **93.5** | **50.9** | **68.7** |

## 5. 한계 (논문이 인정한 것 + 내가 보기에)

- **B 완료 후 원래 A로 돌아가는 재개(resumption)는 다루지 않음** (프로젝트 페이지 기준)
- 접촉 단계 라벨이 붙은 시연으로 **재학습이 필요**함 → 공개 체크포인트에 바로 못 씀
- 저사양 환경(추론 지연)에 대한 논의 없음

## 6. 내 연구와의 관계

- **겹치는 것**: 문제 정의, LIBERO-Goal, 전환 시점을 단계별로 나눠 보는 실험 설계까지 사실상 같음
- **우리 실측과 일치**: π0의 Mid 8.3% / Late 10.2%는, 우리가 SmolVLA로 잰 잡은 직후 25% / 들고 이동 중 39%(flush)와 같은 이야기 — 학습 없이 지시만 바꾸면 무너진다
- **남아 있는 차별점 후보**
  1. **재개(A 다시 하기)** — SwitchVLA가 다루지 않음. 우리 실험에서는 flush 계열이 0~3%로 완전히 실패
  2. **학습 없이(training-free)** 되는 방법 — SwitchVLA는 재학습 필요, 우리 retreat은 공개 체크포인트 그대로 사용
  3. **왜 실패하는지의 설명** — 우리 결과는 "지시를 무시해서"가 아니라 **자세가 학습 분포에서 벗어나서**임을 시사 ([LIBERO-Plus](LIBERO-Plus.md) 와 연결)
  4. 저사양(1080 Ti, 추론 지연) 조건

## 7. 모르는 용어 메모

- behavior modulation: 정책을 새로 학습하는 대신, 조건 입력으로 행동 양식을 바꾸는 것
- contact phase: 물체와 접촉 전/중/후로 나눈 시연 구간 라벨
