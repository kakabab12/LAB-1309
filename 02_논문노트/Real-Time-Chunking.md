# Real-Time Execution of Action Chunking Flow Policies (RTC)

- **저자 / 연도 / 학회**: Kevin Black, Manuel Y. Galliker, Sergey Levine / 2025 / NeurIPS 2025
- **링크**: https://arxiv.org/abs/2506.07339
- **코드**: LeRobot 에 구현 포함 (`lerobot/policies/rtc`, SmolVLA·π0·π0.5 지원)
- **읽은 날짜**: 2026-09-17 (초록·LeRobot 구현 코드 기준)

## 1. 이 논문이 푸는 문제 (한 문장)

action chunk 를 쓰는 VLA 는 다음 chunk 를 계산하는 동안 멈추거나, 이어붙일 때 동작이 튄다.

## 2. 핵심 아이디어 (한 문장)

새 chunk 생성을 **inpainting 문제**로 본다 — 이전 chunk 에서 곧 실행될 동작은 "고정", 나머지는 그에 맞춰 채운다.

## 3. 방법

- flow matching 의 매 denoising 단계에서, 이전 chunk 의 남은 동작과 현재 예측의 차이를 **guidance** 로 더해 보정
- 앞 `d` 스텝(추론 지연만큼)은 가중치 1로 완전 고정, 이후 `execution_horizon` 까지 가중치가 줄어듦
- **재학습 없이** 추론 때만 적용

## 4. 실험 (논문)

- 추론 지연이 커도 성공률이 유지되고, 성냥 켜기 같은 정밀 작업도 가능

## 5. 한계

- 전환(지시가 바뀌는 순간)을 겨냥한 방법은 아님. 같은 지시에서 chunk 를 이어붙이는 문제

## 6. 내 연구와의 관계

- **가져온 것**: 전환 전략 `rtc` 로 구현 (`switch_experiment.py`). 지시를 바꾸는 순간에도 이전 chunk 를 guidance 로 줘서 **리셋 없이 매끄럽게** 이어지게 함 → [2026-09-17 일지](../03_일지/2026-09-17.md)
- 우리 `blend`(가중평균)는 RTC 의 가장 단순한 근사. RTC 는 모델 안에서 이어붙이므로 더 자연스러울 것으로 기대
- 연구주제.md 의 "저사양·추론 지연" 실험에도 그대로 씀 (`--rtc-delay`)

## 7. 용어

- **inpainting**: 일부를 고정하고 나머지를 채워 생성하는 것
- **guidance**: 생성 과정에서 원하는 방향으로 끌어당기는 보정 항
