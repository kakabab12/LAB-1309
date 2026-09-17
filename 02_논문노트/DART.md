# DART: Noise Injection for Robust Imitation Learning

- **저자 / 연도 / 학회**: Michael Laskey, Jonathan Lee, Roy Fox, Anca Dragan, Ken Goldberg / 2017 / CoRL 2017
- **링크**: https://arxiv.org/abs/1703.09327
- **읽은 날짜**: 2026-09-17 (초록 기준)

## 1. 문제

모방학습(BC)은 로봇이 시연에서 조금만 벗어나도 오차가 쌓여 실패한다 (**공변량 이동**).

## 2. 핵심 아이디어

시연을 모을 때 **일부러 노이즈를 넣어** 시연자가 살짝 벗어난 상태를 만들고, 거기서 **되돌아오는 교정 동작**까지 데이터에 담는다.

## 3. 방법

- 시연자(사람 또는 알고리즘) 동작에 노이즈를 더해 실행
- 노이즈 크기는 로봇 정책의 실제 오차에 맞춰 조절
- DAgger 처럼 매번 전문가에게 물어볼 필요가 없음

## 4. 실험

- MuJoCo 에서 on-policy 방법(DAgger)과 비슷한 성능, 실물 Toyota HSR 잡기에서 BC 보다 크게 좋음

## 6. 내 연구와의 관계

- 우리 문제가 정확히 공변량 이동이다 (전환 순간 팔이 12~27cm 벗어남, [ablation](../04_실습/05_libero/retreat_ablation/README.md))
- **가져온 것**: 자세 교란 데이터 수집(`collect_robust_data.py --mode pose`)이 DART 의 "벗어난 상태를 일부러 만든다"는 발상. 3차에서는 **실행 중 동작 노이즈**도 넣어 교정 동작을 모을 계획
- **차이**: DART 는 좋은 시연자가 있지만 우리는 없다 → 모델 스스로 성공한 것만 걸러야 함 (self-imitation 의 한계, [Q-Planning](Q-Planning.md) 참고)
