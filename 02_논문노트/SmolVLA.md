# SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics

- **저자 / 연도**: Mustafa Shukor, Dana Aubakirova 외 (Hugging Face) / 2025 / arXiv
- **링크**: https://arxiv.org/abs/2506.01844 · 블로그 https://huggingface.co/blog/smolvla
- **읽은 날짜**: 2026-09-17 (초록·블로그·LeRobot 코드 기준)

## 1. 문제

VLA 는 너무 커서 학습·추론 비용이 크다.

## 2. 핵심 아이디어

작은 VLM(SmolVLM2) 위에 **flow matching action expert** 를 붙이고, 공개 커뮤니티 데이터로 학습해서 0.45B 로도 큰 모델에 가까운 성능.

## 3. 방법 (LeRobot 코드에서 확인한 것)

| 항목 | 값 |
|---|---|
| VLM | SmolVLM2-500M, 앞쪽 일부 층만 사용 |
| action expert | LlamaModel, VLM 과 cross-attention |
| 출력 | flow matching, **chunk 50스텝** |
| 입력 이미지 | 512×512 로 패딩 리사이즈 |
| LIBERO 체크포인트 설정 | `train_expert_only=True`, `freeze_vision_encoder=True` |
| 비동기 추론 | 인식·예측과 실행을 분리 |

## 6. 내 연구와의 관계

- 사용 모델. `HuggingFaceVLA/smolvla_libero` 체크포인트
- LoRA 를 action expert 에만 붙인 근거가 체크포인트 설정(`train_expert_only`)
- 논문 LIBERO-Goal 점수와 달리 우리 환경 기준 점수는 74% (`n_action_steps=10`)
