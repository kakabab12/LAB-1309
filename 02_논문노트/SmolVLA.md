# SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics

- **저자 / 연도**: Mustafa Shukor, Dana Aubakirova 외 (Hugging Face) / 2025 / arXiv
- **링크**: https://arxiv.org/abs/2506.01844 · 블로그 https://huggingface.co/blog/smolvla
- **코드**: LeRobot (`lerobot/policies/smolvla`)
- **읽은 날짜**: 2026-09-17 (초록·블로그 + LeRobot 코드와 체크포인트 설정을 직접 확인)

---

## 📌 쉬운 말 요약

**한 줄로**: 카메라로 보고, 말을 알아듣고, 로봇팔을 움직이는 모델(VLA)을 **아주 작게(4.5억 파라미터)** 만들어서 **보통 GPU 한 장으로도 학습·실행**할 수 있게 한 모델입니다.

**비유로**: 큰 VLA 가 대형 버스라면 SmolVLA 는 경차입니다. 짐을 조금 덜 싣지만 동네 골목(연구실 GPU, 노트북)을 다닐 수 있습니다. 그리고 "짐칸"과 "운전석"을 분리해서, 눈과 귀(VLM)는 기존 작은 모델을 쓰고 **손 움직임을 담당하는 부분(action expert)** 만 따로 붙였습니다.

**우리 연구와의 관계**: **우리가 쓰는 모델**입니다. 1080 Ti(11GB)에서 돌릴 수 있는 VLA 중 LIBERO 체크포인트가 공개된 것이라 골랐습니다.

---

## 1. 배경

OpenVLA(7B), π0(3B) 같은 VLA 는 성능은 좋지만

- 학습에 대형 GPU 여러 장이 필요하고
- 추론이 느려서 실시간 제어가 어렵고
- 학습 데이터가 비공개인 경우가 많습니다

## 2. 핵심 아이디어

1. **작은 VLM 을 쓰고 더 줄인다** — SmolVLM2 를 쓰되 앞쪽 층만 사용
2. **동작은 별도의 작은 전문가(action expert)가 만든다** — flow matching 으로 동작 묶음 생성
3. **공개 커뮤니티 데이터로 학습** — LeRobot 에 올라온 데이터셋 사용
4. **비동기 추론** — 계산과 실행을 분리해서 반응 속도 향상

## 3. 구조 — 우리가 코드와 체크포인트에서 직접 확인한 것

```
카메라 이미지 2장 ─┐
 (256×256 → 512×512 패딩)
언어 지시 ────────┤──▶  VLM (SmolVLM2-500M, 얼림)  ──▶  특징
로봇 상태 8차원 ───┘                                     │
                                                      (cross-attention)
                                                         ▼
노이즈 ──────────────────────────────────▶  action expert (LlamaModel, 32층)
                                           │  10단계 flow matching
                                           ▼
                                  동작 묶음 50스텝 × 7차원
```

| 항목 | 값 | 확인한 곳 |
|---|---|---|
| 전체 파라미터 | 약 6.1억 (VLM 포함) | 직접 셈 |
| 학습 가능 (체크포인트 설정) | 약 9,750만 (action expert + 투영층) | 직접 셈 |
| action expert | LlamaModel **32층**, 폭은 VLM 의 **0.5배** | 체크포인트 config |
| 어텐션 배치 | **2층마다** self-attention, 나머지는 VLM 에 cross-attention | config `self_attn_every_n_layers=2` |
| 동작 생성 | flow matching **10단계** | config `num_steps=10` |
| chunk 길이 | **50** | config |
| 이미지 | 512×512 로 패딩 리사이즈, SigLIP 입력 범위 [-1, 1] | 코드 |
| 로봇 상태 | 끝단 위치 3 + 축-각 3 + 그리퍼 2 = **8차원** | 코드 |
| 동작 | 끝단 변화량 6 + 그리퍼 1 = **7차원** | 코드 |
| LIBERO 체크포인트 | `train_expert_only=True`, `freeze_vision_encoder=True` | config |

### flow matching 이 뭔가

```
노이즈 ──▶ 조금 덜 노이즈 ──▶ ... (10단계) ... ──▶ 깔끔한 동작 묶음
```

매 단계에서 모델이 "어느 방향으로 다듬어야 하나(속도)"를 예측하고 조금씩 이동합니다. 확산 모델(diffusion)과 비슷하지만 더 곧은 경로로 가도록 학습해서 **적은 단계로도** 동작이 나옵니다. [RTC](Real-Time-Chunking.md) 는 이 10단계 사이사이에 보정을 끼워 넣는 방식입니다.

## 4. 실험 (논문)

- 시뮬레이션 **LIBERO, Meta-World** 와 실물 **SO100, SO101** 로봇
- 10배 큰 VLA 들과 비슷하거나 더 좋은 성능, ACT 보다 좋음
- 비동기 추론으로 반응 약 30% 빠르고 작업 처리량 약 2배

## 5. 한계

- 작아서 복잡하고 긴 태스크에는 한계
- 학습 데이터가 "지시 하나 → 처음부터 끝까지" 형태 → **도중 전환은 학습된 적 없음** (우리 연구 주제)

## 6. 내 연구와의 관계

### 우리 환경 실측

| 항목 | 값 | 기록 |
|---|---|---|
| LIBERO-Goal 기준 점수 | **74%** (`n_action_steps=10`) | [9/14 일지](../03_일지/2026-09-14.md) |
| 1080 Ti 추론 VRAM | 약 2.2GB (fp32) | 〃 |
| LoRA 학습 VRAM | 2.67GB (배치 4) | [LoRA 1차](../04_실습/05_libero/lora_v1/README.md) |

논문의 LIBERO-Goal 점수보다 낮은데, `n_action_steps` 를 체크포인트 기본값(1)과 다르게 10으로 둔 영향일 수 있습니다 (확인 안 됨).

### 이 구조가 우리 결과에 준 영향

- **action expert 만 LoRA** — 체크포인트가 원래 그 부분만 학습하도록 설정돼 있음. 전환 실패가 "지시 이해"가 아니라 "동작 생성" 문제였으므로 맞는 선택
- **chunk 50스텝** → 지시가 바뀌는 순간 **남은 동작을 어떻게 처리하나**(flush/keep/blend/rtc)가 문제가 됨
- **flow matching** → [RTC](Real-Time-Chunking.md) 를 학습 없이 붙일 수 있었고, [Q-Planning](Q-Planning.md) 식으로 **후보를 여러 개** 뽑을 수도 있음
- **로봇 상태에 끝단 위치가 직접 들어감** → 팔 위치가 학습 분포에서 벗어나면 입력 자체가 낯설어짐. [ablation](../04_실습/05_libero/retreat_ablation/README.md)에서 위치 복구가 핵심이었던 것과 연결

## 7. 용어

| 용어 | 뜻 |
|---|---|
| VLA | 이미지 + 언어 → 로봇 동작 모델 |
| VLM | 이미지 + 언어를 이해하는 모델 |
| action expert | VLM 특징을 받아 동작을 만드는 작은 모듈 |
| flow matching | 노이즈를 여러 단계에 걸쳐 데이터로 다듬는 생성 방법 |
| cross-attention | 다른 쪽(여기선 VLM) 정보를 참고하는 어텐션 |
| 비동기 추론 | 동작 계산과 실행을 동시에 따로 돌리는 방식 |
| SigLIP | 이미지와 텍스트를 같은 공간에 놓는 비전 인코더 |
